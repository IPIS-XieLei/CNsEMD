import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from NetModel.CNTSeg_V1 import UNet2D as cntsegunet
from NetModel.CNTSeg_V1 import DatafusionNet_2 as cntsegfusionnet_2
from NetModel.CNTSegV2_Dedicated import CNTSegV2_Dedicated
from NetModel.PHMNet import PHMNet
from CNsEMD_dataloader import CN_MyTrainDataset
from torchvision.transforms import transforms
from CNsEMD_mutilloss import SoftDiceLossSingle, diceCoeff_single, compute_sdf
import os
import random
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import sys
import datetime
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree

original_print = print
def timed_print(*args, **kwargs):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_print(f"[{now}]", *args, **kwargs)
print = timed_print

EPS = 1e-20
DICE_WEIGHT = 0.5
CE_WEIGHT = 0.5
SEED = 3407

def get_parser():
    parser = argparse.ArgumentParser(description="2D UNet Training & Prediction")
    parser.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="num workers")
    parser.add_argument("--epochs", type=int, default=200, help="train epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--model", type=str, default="PHMNet",
                        choices=["CNTSeg_V1_T1DEC", "CNTSegV2_Dedicated", "PHMNet"])

    parser.add_argument("--raw_data_root", type=str, default="/path/to/CNsEMD")
    parser.add_argument("--fornetwork_root", type=str, default="/path/to/CNsEMD_preprocessed_data")
    parser.add_argument("--result_root", type=str, default="/path/to/CNsEMD_results")
    parser.add_argument("--split_json", type=str, default="/path/to/CNsEMD/splits_final_train.json")
    parser.add_argument("--crop_h", type=int, default=128)
    parser.add_argument("--crop_w", type=int, default=160)
    parser.add_argument("--folds", type=str, default="fold_0,fold_1,fold_2,fold_3,fold_4")#fold_0,fold_1,fold_2,fold_3,fold_4
    return parser.parse_args()

def calculate_metrics(pre_path, gt_case_path, file_prefix):
    gt_all_path = os.path.join(
        gt_case_path,
        f"{file_prefix}-label.nii.gz"
    )

    gt_img = nib.load(gt_all_path)
    gt_all = gt_img.get_fdata().astype(np.int16)
    spacing = np.asarray(
        nib.affines.voxel_sizes(gt_img.affine),
        dtype=np.float64
    )

    pre_img = nib.load(pre_path)
    pre = pre_img.get_fdata().astype(np.int16)

    if pre.shape != gt_all.shape:
        raise ValueError(
            f"{file_prefix}: prediction shape {pre.shape} "
            f"does not match GT shape {gt_all.shape}."
        )

    def get_empty_penalty(mask_shape, voxel_spacing):
        shape = np.asarray(
            mask_shape,
            dtype=np.float64
        )

        voxel_spacing = np.asarray(
            voxel_spacing,
            dtype=np.float64
        )

        physical_extent = (
            shape - 1.0
        ) * voxel_spacing

        return float(
            np.linalg.norm(physical_extent)
        )

    def check_empty_mask(gt, pred, class_id):
        if gt.sum() == 0:
            raise ValueError(
                f"{file_prefix}: ground-truth mask "
                f"for class {class_id} is empty."
            )

        if pred.sum() == 0:
            return get_empty_penalty(
                gt.shape,
                spacing
            )

        return None

    def compute_asd(gt, pred, class_id):
        penalty = check_empty_mask(
            gt,
            pred,
            class_id
        )

        if penalty is not None:
            return penalty

        gt_boundary = find_boundaries(
            gt,
            mode="inner"
        ).astype(bool)

        pred_boundary = find_boundaries(
            pred,
            mode="inner"
        ).astype(bool)

        gt_points = (
            np.argwhere(gt_boundary)
            * spacing
        )

        pred_points = (
            np.argwhere(pred_boundary)
            * spacing
        )

        if (
            len(gt_points) == 0
            or len(pred_points) == 0
        ):
            raise RuntimeError(
                f"{file_prefix}: boundary extraction "
                f"failed for class {class_id}."
            )

        gt_tree = cKDTree(gt_points)
        pred_tree = cKDTree(pred_points)

        pred_to_gt = gt_tree.query(
            pred_points
        )[0]

        gt_to_pred = pred_tree.query(
            gt_points
        )[0]

        asd = (
            pred_to_gt.sum()
            + gt_to_pred.sum()
        ) / (
            len(pred_points)
            + len(gt_points)
        )

        asd = float(asd)

        if not np.isfinite(asd):
            raise RuntimeError(
                f"{file_prefix}: non-finite ASD "
                f"for class {class_id}."
            )

        return asd

    def compute_ahd(gt, pred, class_id):
        penalty = check_empty_mask(
            gt,
            pred,
            class_id
        )

        if penalty is not None:
            return penalty

        gt_itk = sitk.GetImageFromArray(
            gt.transpose(2, 1, 0).astype(np.uint8)
        )

        pred_itk = sitk.GetImageFromArray(
            pred.transpose(2, 1, 0).astype(np.uint8)
        )

        itk_spacing = tuple(
            float(value)
            for value in spacing
        )

        gt_itk.SetSpacing(itk_spacing)
        pred_itk.SetSpacing(itk_spacing)

        hd_filter = (
            sitk.HausdorffDistanceImageFilter()
        )

        hd_filter.Execute(
            gt_itk,
            pred_itk
        )

        ahd = float(
            hd_filter.GetAverageHausdorffDistance()
        )

        if not np.isfinite(ahd):
            raise RuntimeError(
                f"{file_prefix}: non-finite AHD "
                f"for class {class_id}."
            )

        return ahd

    def compute_overlap_metrics(gt, pred):
        tp = np.logical_and(
            gt == 1,
            pred == 1
        ).sum()

        fp = np.logical_and(
            gt == 0,
            pred == 1
        ).sum()

        fn = np.logical_and(
            gt == 1,
            pred == 0
        ).sum()

        dice = (
            2.0 * tp
        ) / (
            2.0 * tp
            + fp
            + fn
            + EPS
        )

        jac = tp / (
            tp
            + fp
            + fn
            + EPS
        )

        return (
            float(dice),
            float(jac)
        )

    metrics = []

    for class_id in [1, 2, 3, 4]:
        gt_class = (
            gt_all == class_id
        ).astype(np.uint8)

        pred_class = (
            pre == class_id
        ).astype(np.uint8)

        dice, jac = compute_overlap_metrics(
            gt_class,
            pred_class
        )

        asd = compute_asd(
            gt_class,
            pred_class,
            class_id
        )

        ahd = compute_ahd(
            gt_class,
            pred_class,
            class_id
        )

        # 每类指标顺序：
        # Dice、Jac、ASD、AHD
        metrics.extend([
            dice,
            jac,
            asd,
            ahd
        ])

    metrics = np.asarray(
        metrics,
        dtype=np.float64
    )

    if not np.isfinite(metrics).all():
        bad_indices = np.where(
            ~np.isfinite(metrics)
        )[0]

        raise RuntimeError(
            f"{file_prefix}: non-finite metrics "
            f"at indices {bad_indices.tolist()}."
        )

    return metrics.tolist()

def predict_and_evaluate(model, val_patch_path, save_path, case_list, args, device):
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    all_metrics = []
    log_file = os.path.join(save_path, "metrics_all.txt")
    with open(log_file, "w") as f:
        f.write("=" * 120 + "\n")
        f.write(
            "Case\t"
            "Dice_ON\tJac_ON\tASD_ON\tAHD_ON\t"
            "Dice_OCN\tJac_OCN\tASD_OCN\tAHD_OCN\t"
            "Dice_TGN\tJac_TGN\tASD_TGN\tAHD_TGN\t"
            "Dice_FVN\tJac_FVN\tASD_FVN\tAHD_FVN\n"
        )
        f.write("=" * 120 + "\n")

    def get_auto_crop_coords(vol_shape, target=(128, 160, 128)):
        coords = []
        for dim in range(3):
            raw = vol_shape[dim]
            tgt = target[dim]
            start = (raw - tgt) // 2
            end = start + tgt
            coords.append((start, end))
        return coords

    x_t1_dir = os.path.join(val_patch_path, "x_t1_data")
    x_dec_dir = os.path.join(val_patch_path, "x_dec_data")
    y_label_dir = os.path.join(val_patch_path, "y_label_data")
    test_dataset = CN_MyTrainDataset(
        x_t1_dir, x_dec_dir, y_label_dir,
        x_transform=transforms.ToTensor(),
        y_transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pred_cache = {}
    with torch.no_grad():
        for x_t1, x_dec, *rest in test_loader:
            names = rest[-1]
            x_t1, x_dec, = x_t1.to(device), x_dec.to(device)
            ################################################################
            if args.model == "CNTSeg_V1_T1DEC":
                with torch.no_grad():
                    feat1 = model_t1(x_t1)
                    feat2 = model_dec(x_dec)
                out = model(feat1, feat2)
            elif args.model == "CNTSegV2_Dedicated":
                out, _ = model(x_t1, x_dec)
            else:
                out = model(x_t1, x_dec)
            ################################################################
            out = torch.argmax(out, dim=1)
            pred_batch = out.cpu().numpy().astype(np.uint8)
            for i in range(len(names)):
                name = names[i]
                slice_pred = pred_batch[i]
                pred_cache[name] = slice_pred

    for case in case_list:
        file_prefix = case.replace("CNs_", "")
        raw_t1_path = os.path.join(args.raw_data_root, "T1", f"{file_prefix}-T1.nii.gz")
        raw_img = nib.load(raw_t1_path)
        original_shape = raw_img.shape
        original_affine = raw_img.affine
        (h_start, h_end), (w_start, w_end), (d_start, d_end) = get_auto_crop_coords(original_shape)
        pred_vol = np.zeros(original_shape, dtype=np.uint8)
        for name, slice_pred in pred_cache.items():
            if name.startswith(file_prefix):
                z_cropped = int(name.split("_z")[-1])
                z_original = d_start + z_cropped
                if 0 <= z_original < original_shape[2]:
                    pred_vol[h_start:h_end, w_start:w_end, z_original] = slice_pred
        pred_path = os.path.join(save_path, f"{file_prefix}_pred.nii.gz")
        nib.save(nib.Nifti1Image(pred_vol, original_affine), pred_path)
        print(f"predicting {file_prefix}, shape {original_shape}")
        gt_case_path = os.path.join(args.raw_data_root, "GT")
        metrics = calculate_metrics(pred_path, gt_case_path, file_prefix)
        all_metrics.append(metrics)
        with open(log_file, 'a') as f:
            f.write(
                f"{file_prefix}\t" + "\t".join([f"{v:.6f}" if not np.isnan(v) else "nan" for v in metrics]) + "\n")
    if all_metrics:
        arr = np.asarray(
            all_metrics,
            dtype=np.float64
        )

        if not np.isfinite(arr).all():
            bad_locations = np.argwhere(
                ~np.isfinite(arr)
            )

            raise RuntimeError(
                "Non-finite validation metrics detected at "
                f"{bad_locations.tolist()}."
            )

        mean_vals = np.mean(
            arr,
            axis=0
        )

        std_vals = np.std(
            arr,
            axis=0,
            ddof=0
        )

        mean_std_str = [
            f"{mean:.4f}±{std:.4f}"
            for mean, std
            in zip(mean_vals, std_vals)
        ]

        print("\n" + "=" * 80)
        print("FINAL SUMMARY (MEAN ± STD)")
        print("=" * 80)

        print(
            f"Mean Dice: "
            f"ON={mean_vals[0]:.4f}, "
            f"OCN={mean_vals[4]:.4f}, "
            f"TGN={mean_vals[8]:.4f}, "
            f"FVN={mean_vals[12]:.4f}"
        )

        mean_cns_dice = np.mean([
            mean_vals[0],
            mean_vals[4],
            mean_vals[8],
            mean_vals[12]
        ])

        print(
            f"Mean Dice: CNs={mean_cns_dice:.4f}"
        )
        print("=" * 80)

def compute_val_dice(model, val_patch_path, case_list, args, device):
    model.eval()
    pred_slices = {case: {} for case in case_list}
    gt_slices = {case: {} for case in case_list}

    x_t1_dir = os.path.join(val_patch_path, "x_t1_data")
    x_dec_dir = os.path.join(val_patch_path, "x_dec_data")
    y_label_dir = os.path.join(val_patch_path, "y_label_data")
    dataset = CN_MyTrainDataset(
        x_t1_dir, x_dec_dir, y_label_dir,
        x_transform=transforms.ToTensor(),
        y_transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        for x_t1, x_dec, y_bg, y_on, y_ocn, y_tgn, y_fvn, names in loader:
            gt = torch.argmax(torch.cat([y_bg, y_on, y_ocn, y_tgn, y_fvn], 1), dim=1).to(device)
            x_t1, x_dec = x_t1.to(device), x_dec.to(device)
            if args.model == "CNTSeg_V1_T1DEC":
                with torch.no_grad():
                    feat1 = model_t1(x_t1)
                    feat2 = model_dec(x_dec)
                out = model(feat1, feat2)
            elif args.model == "CNTSegV2_Dedicated":
                out, _ = model(x_t1, x_dec)
            else:
                out = model(x_t1, x_dec)
            pred = torch.argmax(out, dim=1).cpu().numpy().astype(np.uint8)
            gt_np = gt.cpu().numpy().astype(np.uint8)
            for i, name in enumerate(names):
                case = name.split("_z")[0]
                z = int(name.split("_z")[1])
                pred_slices[case][z] = pred[i]
                gt_slices[case][z] = gt_np[i]

    case_dices = []
    for case in case_list:
        if not pred_slices[case]:
            continue
        z_list = sorted(pred_slices[case].keys())

        H, W = pred_slices[case][z_list[0]].shape
        pred_vol = np.zeros((H, W, len(z_list)), dtype=np.uint8)
        gt_vol = np.zeros((H, W, len(z_list)), dtype=np.uint8)
        for i, z in enumerate(z_list):
            pred_vol[..., i] = pred_slices[case][z]
            gt_vol[..., i] = gt_slices[case][z]

        dice_per_class = []
        for c in range(1, 5):
            pred_c = (pred_vol == c)
            gt_c = (gt_vol == c)
            inter = np.logical_and(pred_c, gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice = (2.0 * inter) / (union + 1e-8)
            dice_per_class.append(dice)
        case_dices.append(dice_per_class)

    if not case_dices:
        return 0.0, (0.0, 0.0, 0.0, 0.0)

    mean_dice_per_class = np.mean(case_dices, axis=0)
    mean_dice_avg = np.mean(mean_dice_per_class)
    return mean_dice_avg, tuple(mean_dice_per_class)

def train(fold, args, device, model):
    fold_save_root = os.path.join(args.result_root, f"{args.model}", fold)
    base = os.path.join(args.fornetwork_root, fold, "train")
    val_patch_path = os.path.join(args.fornetwork_root, fold, "val")
    os.makedirs(fold_save_root, exist_ok=True)
    original_stdout = sys.stdout
    log_file_path = os.path.join(fold_save_root, "training_log.txt")

    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = original_stdout
            self.log = open(filename, "a", buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger(log_file_path)
    print(f"Log saved to: {log_file_path}")
    print("=" * 50)
    print("           ✅ Training Parameters")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg:20s} = {value}")
    print("=" * 50)

    train_dataset = CN_MyTrainDataset(
        os.path.join(base, "x_t1_data"),
        os.path.join(base, "x_dec_data"),
        os.path.join(base, "y_label_data"),
        x_transform=transforms.ToTensor(),
        y_transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_ce = nn.CrossEntropyLoss().to(device)
    loss_dice = SoftDiceLossSingle(5).to(device)
    losses_dis = nn.SmoothL1Loss().to(device)  ##for CNTSegV2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6
    )

    best_dice = -1
    best_epoch = 1
    patience = 20
    counter = 0
    early_stop = False

    def get_cases(folder):
        cases = []
        for f in os.listdir(os.path.join(folder, "x_t1_data")):
            if "_z" in f:
                case = f.split("_z")[0]
                if case not in cases:
                    cases.append(case)
        return cases
    val_cases = get_cases(val_patch_path)
    print('-' * 30)
    print('Training start... ')
    print('-' * 30)

    for epoch in range(args.epochs):
        if early_stop:
            break
        model.train()
        stats = np.zeros(6)
        total = len(train_loader.dataset)
        for step, (x_t1, x_dec, y_bg, y_on, y_ocn, y_tgn, y_fvn, _) in enumerate(train_loader):
            gt = torch.argmax(torch.cat([y_bg, y_on, y_ocn, y_tgn, y_fvn], 1), dim=1).to(device)
            x_t1, x_dec = x_t1.to(device), x_dec.to(device)
            if args.model == "CNTSeg_V1_T1DEC":
                with torch.no_grad():
                    feat1 = model_t1(x_t1)
                    feat2 = model_dec(x_dec)
                outputs = model(feat1, feat2)
                loss_ce_val = loss_ce(outputs, gt)
                loss_dice_val = loss_dice(outputs, gt)
                loss = CE_WEIGHT * loss_ce_val + DICE_WEIGHT * loss_dice_val
            elif args.model == "CNTSegV2_Dedicated":
                outputs, pre_dis = model(x_t1, x_dec)
                groundtruth1, groundtruth2, groundtruth3, groundtruth4 = y_on.to(device), y_ocn.to(device), y_tgn.to(
                    device), y_fvn.to(device)
                binary_gt = groundtruth1 + groundtruth2 + groundtruth3 + groundtruth4
                binary_gt[binary_gt > 0] = 1
                with torch.no_grad():
                    dis = torch.from_numpy(
                        compute_sdf(binary_gt.cpu().numpy(), pre_dis.shape)).float().to(device)
                losses_dis_val = losses_dis(pre_dis, dis)
                loss_ce_val = loss_ce(outputs, gt)
                loss_dice_val = loss_dice(outputs, gt)
                loss = CE_WEIGHT * loss_ce_val + DICE_WEIGHT * loss_dice_val + 0.5 * losses_dis_val
            else:
                outputs = model(x_t1, x_dec)
                loss_ce_val = loss_ce(outputs, gt)
                loss_dice_val = loss_dice(outputs, gt)
                loss = CE_WEIGHT * loss_ce_val + DICE_WEIGHT * loss_dice_val
            ################################################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_dice = diceCoeff_single(outputs, gt, num_classes=5)
                d1, d2, d3, d4 = train_dice
                m_dice = np.mean(train_dice)
            stats += [loss.item(), m_dice, d1, d2, d3, d4]
            current = min((step + 1) * args.batch_size, total)
            current_lr = optimizer.param_groups[0]['lr']
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"\rEpoch:{epoch + 1}/{args.epochs} [{'=' * int(20 * current / total):<20}] ON:{d1:.3f} OCN:{d2:.3f} TGN:{d3:.3f} FVN:{d4:.3f} LR:{current_lr:.6f} {current}/{total}"
            try:
                sys.stdout.terminal.write(msg)
                sys.stdout.terminal.flush()
            except Exception as e:
                print(f"\nTerminal write error: {e}")
                sys.stdout.write(msg)
        print()
        val_mean, (val_d1, val_d2, val_d3, val_d4) = compute_val_dice(model, val_patch_path, val_cases, args, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch:{epoch + 1:2d}/{args.epochs} | Loss:{stats[0] / len(train_loader):.3f} | ValDice:{val_mean:.4f} | LR:{current_lr:.6f}")
        scheduler.step(val_mean)
        if val_mean > best_dice:
            print(
                f"Yayy! New best meanDice: {val_mean:.4f} | Val:[{val_d1:.3f}, {val_d2:.3f}, {val_d3:.3f}, {val_d4:.3f}]")
            best_dice = val_mean
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), os.path.join(fold_save_root, "BEST_MODEL.pth"))
        else:
            counter += 1
            print(f"No improvement")
            if counter >= patience:
                print(f"\nEarly STOP triggered! Best epoch: {best_epoch}, Best Dice: {best_dice:.4f}")
                early_stop = True
                model.load_state_dict(torch.load(os.path.join(fold_save_root, "BEST_MODEL.pth")))
                break
    print("\n" + "-" * 50)
    print("Training complete.")
    print(f"Best epoch: {best_epoch}, Best Dice: {best_dice:.4f}")
    model.load_state_dict(torch.load(os.path.join(fold_save_root, "BEST_MODEL.pth")))
    predict_save_path = os.path.join(fold_save_root, "Predictions")
    predict_and_evaluate(model, val_patch_path, predict_save_path, val_cases, args, device)
    sys.stdout = original_stdout

if __name__ == '__main__':
    args = get_parser()
    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")

    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    fold_list = args.folds.split(",")
    for fold in fold_list:
        set_seed()
        print(f'\n========== TRAIN {fold} ==========')
        torch.cuda.empty_cache()
        ##################################################################################
        if args.model == "PHMNet":
            model = PHMNet().to(device)
        ##################################################################################
        elif args.model == "CNTSeg_V1_T1DEC":
            t1_model_path = os.path.join(args.result_root, "CNTSeg_V1_T1DEC", "UnetT1", fold, "BEST_MODEL.pth")
            dec_model_path = os.path.join(args.result_root, "CNTSeg_V1_T1DEC", "UnetDEC",  fold, "BEST_MODEL.pth")
            model_t1 = cntsegunet(1, 5).to(device)
            model_dec = cntsegunet(3, 5).to(device)
            model_t1.load_state_dict(torch.load(t1_model_path, map_location=device))
            model_dec.load_state_dict(torch.load(dec_model_path, map_location=device))
            model_t1.eval()
            model_dec.eval()
            for param in model_t1.parameters(): param.requires_grad = False
            for param in model_dec.parameters(): param.requires_grad = False
            model = cntsegfusionnet_2(10, 5).to(device)
        elif args.model == "CNTSegV2_Dedicated":
            model = CNTSegV2_Dedicated(1, 3, 5).to(device)
        ##################################################################################
        train(fold, args, device, model)
    print("🎉 ALL DONE!")
