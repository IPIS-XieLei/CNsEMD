#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run five-fold ensemble inference for one T1w--DEC subject.

The preprocessing and checkpoint layout follow ``CNsEMD_test.py``.
The T1w and DEC images must already be co-registered and have identical
spatial geometry. The predicted label map is restored to the original T1w
shape and saved with its affine and header.
"""

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MODEL_CHOICES = ("PHMNet", "CNTSeg_V1_T1DEC", "CNTSegV2_Dedicated")
TARGET_SHAPE = (128, 160, 128)
NUM_CLASSES = 5

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict one CN label map from co-registered T1w and DEC NIfTI images."
    )
    parser.add_argument("--t1", required=True, help="Path to T1.nii or T1.nii.gz")
    parser.add_argument("--dec", required=True, help="Path to DEC.nii or DEC.nii.gz")
    parser.add_argument(
        "--model",
        default="PHMNet",
        choices=MODEL_CHOICES,
        help="Model family used for inference (default: PHMNet).",
    )
    parser.add_argument(
        "--result-root",
        required=True,
        help=(
            "Path to model checkpoints "
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output label path. Default: <T1 stem>-<model>-prediction.nii.gz",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold indices to ensemble (default: 0 1 2 3 4).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda:0 or cpu.",
    )
    return parser


def normalize_t1(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(image, (1, 99))
    image = np.clip(image, low, high)
    return ((image - image.mean()) / (image.std() + 1e-8)).astype(np.float32)


def clean_dec(image: np.ndarray) -> np.ndarray:
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def canonicalize_dec(dec: np.ndarray) -> np.ndarray:
    """Return DEC data in (X, Y, Z, 3) order."""
    if dec.ndim != 4 or dec.shape[-1] != 3:
        raise ValueError(
            "DEC must use channel-last NIfTI layout (X, Y, Z, 3); "
            f"received {dec.shape}."
        )
    return dec


def center_crop_or_pad(
    array: np.ndarray, target: Sequence[int] = TARGET_SHAPE
) -> Tuple[np.ndarray, Tuple[slice, slice, slice], Tuple[slice, slice, slice]]:
    """Center crop/pad three spatial axes and return source/destination slices."""
    if array.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D or channel-last 4D array, got {array.shape}.")

    output_shape = tuple(target) + tuple(array.shape[3:])
    output = np.zeros(output_shape, dtype=array.dtype)
    source_slices = []
    target_slices = []

    for size, wanted in zip(array.shape[:3], target):
        source_start = max((size - wanted) // 2, 0)
        length = min(size, wanted)
        target_start = max((wanted - size) // 2, 0)
        source_slices.append(slice(source_start, source_start + length))
        target_slices.append(slice(target_start, target_start + length))

    src = tuple(source_slices)
    dst = tuple(target_slices)
    if array.ndim == 4:
        output[dst + (slice(None),)] = array[src + (slice(None),)]
    else:
        output[dst] = array[src]
    return output, src, dst


class SingleCaseDataset(Dataset):
    def __init__(self, t1: np.ndarray, dec: np.ndarray) -> None:
        self.t1 = t1
        self.dec = dec

    def __len__(self) -> int:
        return self.t1.shape[2]

    def __getitem__(self, z: int):
        # Match the original code: normalize T1w independently per axial slice.
        t1_slice = normalize_t1(self.t1[:, :, z])
        dec_slice = clean_dec(self.dec[:, :, z, :])
        t1_tensor = torch.from_numpy(t1_slice[None, ...])
        dec_tensor = torch.from_numpy(np.moveaxis(dec_slice, -1, 0))
        return t1_tensor, dec_tensor, z


def load_state_dict(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint content in {path}")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


CNTSegV1Model = Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]
LoadedModel = Union[torch.nn.Module, CNTSegV1Model]


def load_models(
    model_name: str,
    result_root: Path,
    folds: Sequence[int],
    device: torch.device,
) -> List[LoadedModel]:
    """Load the requested fold ensemble using the original directory layout."""
    if model_name == "PHMNet":
        from NetModel.PHMNet import PHMNet
    elif model_name == "CNTSegV2_Dedicated":
        from NetModel.CNTSegV2_Dedicated import CNTSegV2_Dedicated
    else:
        from NetModel.CNTSeg_V1 import DatafusionNet_2, UNet2D

    models: List[LoadedModel] = []
    for fold_index in folds:
        fold = f"fold_{fold_index}"
        if model_name == "PHMNet":
            model = PHMNet(1, 3, NUM_CLASSES).to(device)
            checkpoint = result_root / model_name / fold / "BEST_MODEL.pth"
            model.load_state_dict(load_state_dict(checkpoint, device), strict=False)
            models.append(freeze(model))
        elif model_name == "CNTSegV2_Dedicated":
            model = CNTSegV2_Dedicated(1, 3, NUM_CLASSES).to(device)
            checkpoint = result_root / model_name / fold / "BEST_MODEL.pth"
            model.load_state_dict(load_state_dict(checkpoint, device), strict=False)
            models.append(freeze(model))
        else:
            t1_model = UNet2D(1, NUM_CLASSES).to(device)
            dec_model = UNet2D(3, NUM_CLASSES).to(device)
            fusion_model = DatafusionNet_2(10, NUM_CLASSES).to(device)

            t1_checkpoint = (
                result_root
                / model_name
                / "UnetT1"
                / fold
                / "BEST_MODEL.pth"
            )
            dec_checkpoint = (
                result_root
                / model_name
                / "UnetDEC"
                / fold
                / "BEST_MODEL.pth"
            )
            fusion_checkpoint = result_root / model_name / fold / "BEST_MODEL.pth"
            t1_model.load_state_dict(load_state_dict(t1_checkpoint, device))
            dec_model.load_state_dict(load_state_dict(dec_checkpoint, device))
            fusion_model.load_state_dict(
                load_state_dict(fusion_checkpoint, device), strict=False
            )
            models.append(
                (freeze(t1_model), freeze(dec_model), freeze(fusion_model))
            )
    return models


def extract_logits(output) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        shape = getattr(output, "shape", None)
        raise RuntimeError(f"Expected BxCxHxW logits, received {shape!r}.")
    return output


def forward_model(
    model_name: str,
    model: LoadedModel,
    t1: torch.Tensor,
    dec: torch.Tensor,
) -> torch.Tensor:
    if model_name == "CNTSeg_V1_T1DEC":
        t1_model, dec_model, fusion_model = model
        output = fusion_model(t1_model(t1), dec_model(dec))
    else:
        output = model(t1, dec)
    return extract_logits(output)


def default_output_path(t1_path: Path, model_name: str) -> Path:
    name = t1_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return t1_path.with_name(f"{name}-{model_name}-prediction.nii.gz")


def validate_inputs(t1_img: nib.Nifti1Image, dec_img: nib.Nifti1Image) -> None:
    if len(t1_img.shape) != 3:
        raise ValueError(f"T1w must be 3D, but received {t1_img.shape}.")
    dec_shape = dec_img.shape
    if len(dec_shape) != 4 or dec_shape[-1] != 3:
        raise ValueError(
            "DEC must use channel-last NIfTI layout (X, Y, Z, 3); "
            f"received {dec_shape}."
        )
    dec_spatial_shape = dec_shape[:3]
    if tuple(t1_img.shape) != tuple(dec_spatial_shape):
        raise ValueError(
            "T1w and DEC spatial shapes differ: "
            f"{t1_img.shape} versus {dec_spatial_shape}. Register/resample them first."
        )
    if not np.allclose(t1_img.affine, dec_img.affine, rtol=1e-4, atol=1e-3):
        raise ValueError(
            "T1w and DEC affines differ. Register/resample DEC to the T1w space first."
        )


def predict(args: argparse.Namespace) -> Path:
    t1_path = Path(args.t1).expanduser().resolve()
    dec_path = Path(args.dec).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(t1_path, args.model)
    )
    if not t1_path.is_file() or not dec_path.is_file():
        raise FileNotFoundError(f"Input not found: T1={t1_path}, DEC={dec_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")

    print(f"Device: {device}")
    print(f"Model: {args.model}; folds: {args.folds}")
    t1_img = nib.load(str(t1_path))
    dec_img = nib.load(str(dec_path))
    validate_inputs(t1_img, dec_img)

    t1_data = np.asarray(t1_img.dataobj, dtype=np.float32)
    dec_data = canonicalize_dec(np.asarray(dec_img.dataobj, dtype=np.float32))
    t1_crop, source_slices, target_slices = center_crop_or_pad(t1_data)
    dec_crop, dec_source_slices, dec_target_slices = center_crop_or_pad(dec_data)
    if source_slices != dec_source_slices or target_slices != dec_target_slices:
        raise RuntimeError("Internal T1w/DEC crop mismatch.")

    dataset = SingleCaseDataset(t1_crop, dec_crop)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    models = load_models(args.model, result_root, args.folds, device)
    if not models:
        raise RuntimeError("No model folds were loaded.")

    logits = np.zeros((NUM_CLASSES,) + TARGET_SHAPE, dtype=np.float32)
    with torch.no_grad():
        for t1_batch, dec_batch, z_batch in loader:
            t1_batch = t1_batch.to(device, non_blocking=True)
            dec_batch = dec_batch.to(device, non_blocking=True)
            batch_sum = None
            for model in models:
                fold_logits = forward_model(args.model, model, t1_batch, dec_batch)
                batch_sum = fold_logits if batch_sum is None else batch_sum + fold_logits
            batch_logits = (batch_sum / len(models)).cpu().numpy()
            for index, z in enumerate(z_batch.tolist()):
                logits[:, :, :, z] = batch_logits[index]

    prediction_crop = np.argmax(logits, axis=0).astype(np.uint8)
    prediction = np.zeros(t1_img.shape, dtype=np.uint8)
    prediction[source_slices] = prediction_crop[target_slices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = t1_img.header.copy()
    header.set_data_dtype(np.uint8)
    output_img = nib.Nifti1Image(prediction, t1_img.affine, header=header)
    output_img.set_qform(t1_img.get_qform(), int(t1_img.header["qform_code"]))
    output_img.set_sform(t1_img.get_sform(), int(t1_img.header["sform_code"]))
    nib.save(output_img, str(output_path))
    print(f"Saved prediction: {output_path}")
    return output_path


def main() -> None:
    args = get_parser().parse_args()
    if not args.folds or any(fold < 0 for fold in args.folds):
        raise ValueError("--folds must contain one or more non-negative integers.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative.")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    predict(args)


if __name__ == "__main__":
    main()
