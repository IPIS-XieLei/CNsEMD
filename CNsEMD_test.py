import argparse
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from NetModel.CNTSeg_V1 import UNet2D as cntsegunet
from NetModel.CNTSeg_V1 import DatafusionNet_2 as cntsegfusionnet_2
from NetModel.CNTSegV2_Dedicated import CNTSegV2_Dedicated
from NetModel.PHMNet import PHMNet
import SimpleITK as sitk
from tqdm import tqdm
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
EPS = 1e-20

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--model", type=str, default="PHMNet",
                        choices=["CNTSeg_V1_T1DEC", "CNTSegV2_Dedicated", "PHMNet"])
    parser.add_argument("--mode", type=str, default="batch", choices=["single", "batch"])
    parser.add_argument("--input_folder", type=str, default="/path/to/CNsEMD_preprocessed_data/test_set")
    parser.add_argument("--result_root", type=str, default="/path/to/CNsEMD_weights")
    parser.add_argument("--gt_folder", type=str, default="/path/to/CNsEMD_preprocessed_data/test_set/y_label_data")
    parser.add_argument("--suffix", type=str, default=".nii.gz")
    return parser.parse_args()

def normalize_t1(arr):
    arr = np.nan_to_num(arr)
    low = np.percentile(arr, 1)
    high = np.percentile(arr, 99)
    arr = np.clip(arr, low, high)
    mean = arr.mean()
    std = arr.std() + 1e-8
    return ((arr - mean) / std).astype(np.float32)

def dec_keep_original(arr):
    arr = np.nan_to_num(arr)
    return arr.astype(np.float32)

def get_auto_crop_coords(vol_shape, target=(128, 160, 128)):
    coords = []
    for dim in range(3):
        raw = vol_shape[dim]
        tgt = target[dim]
        start = (raw - tgt) // 2
        end = start + tgt
        coords.append((start, end))
    return coords

class CN_InferDataset(Dataset):
    def __init__(self, data_crop_t1, data_crop_dec):
        self.data_t1 = data_crop_t1
        self.data_dec = data_crop_dec
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.data_t1.shape[2]

    def __getitem__(self, z):
        # T1
        t1_slice = self.data_t1[..., z]
        t1_slice = normalize_t1(t1_slice)
        # DEC
        dec_slice = self.data_dec[..., z, :]
        dec_slice = dec_keep_original(dec_slice)

        t1_tensor = self.transform(t1_slice).float()
        dec_tensor = self.transform(dec_slice).float()

        return t1_tensor, dec_tensor, z
def get_empty_penalty(mask_shape, spacing):
    shape = np.asarray(mask_shape, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    return float(np.linalg.norm((shape - 1) * spacing))
def compute_all_metrics(pred, gt, spacing):
    classes = [1, 2, 3, 4]
    metrics_list = []
    def compute_asd(gt_bin, pred_bin, spacing):

        try:
            if gt_bin.sum() == 0:
                raise ValueError("Ground-truth mask is empty.")

            if pred_bin.sum() == 0:
                return get_empty_penalty(gt_bin.shape, spacing)

            gt_boundary = find_boundaries(gt_bin, mode='inner').astype(bool)
            pred_boundary = find_boundaries(pred_bin, mode='inner').astype(bool)

            gt_pts = np.argwhere(gt_boundary) * spacing
            pred_pts = np.argwhere(pred_boundary) * spacing
            if len(gt_pts) == 0 or len(pred_pts) == 0:
                raise RuntimeError(
                    "Boundary extraction failed for a non-empty mask."
                )

            tree_gt = cKDTree(gt_pts)
            tree_pred = cKDTree(pred_pts)
            dist_gt_to_pred = tree_gt.query(pred_pts)[0]
            dist_pred_to_gt = tree_pred.query(gt_pts)[0]
            asd = (dist_gt_to_pred.sum() + dist_pred_to_gt.sum()) / (len(gt_pts) + len(pred_pts))
            return asd
        except Exception as error:
            raise RuntimeError("Distance metric calculation failed.") from error

    def compute_ahd(gt_bin, pred_bin, spacing):
        try:
            if gt_bin.sum() == 0:
                raise ValueError("Ground-truth mask is empty.")

            if pred_bin.sum() == 0:
                return get_empty_penalty(gt_bin.shape, spacing)

            gt_itk = sitk.GetImageFromArray(gt_bin.transpose(2, 1, 0).astype(np.int16))
            gt_itk.SetSpacing(spacing)
            pd_itk = sitk.GetImageFromArray(pred_bin.transpose(2, 1, 0).astype(np.int16))
            pd_itk.SetSpacing(spacing)
            hd = sitk.HausdorffDistanceImageFilter()
            hd.Execute(gt_itk, pd_itk)
            return hd.GetAverageHausdorffDistance()
        except Exception as error:
            raise RuntimeError("Distance metric calculation failed.") from error

    dice_list = []
    jac_list = []
    asd_list = []
    ahd_list = []

    for c in classes:
        gt_bin = (gt == c).astype(np.float32)
        pred_bin = (pred == c).astype(np.float32)

        # Dice, Jaccard
        tp = ((gt_bin == 1) & (pred_bin == 1)).sum()
        fp = ((gt_bin == 0) & (pred_bin == 1)).sum()
        fn = ((gt_bin == 1) & (pred_bin == 0)).sum()
        dice = (2 * tp) / (2 * tp + fp + fn + EPS)
        jac = tp / (tp + fp + fn + EPS)

        # ASD, AHD
        asd = compute_asd(gt_bin, pred_bin, spacing)
        ahd = compute_ahd(gt_bin, pred_bin, spacing)

        dice_list.append(dice)
        jac_list.append(jac)
        asd_list.append(asd)
        ahd_list.append(ahd)

    metrics_list.append(np.mean(dice_list))
    metrics_list.append(np.mean(jac_list))
    metrics_list.append(np.mean(asd_list))
    metrics_list.append(np.mean(ahd_list))

    metrics_list.extend(dice_list)   # Dice per class
    metrics_list.extend(jac_list)    # Jac per class
    metrics_list.extend(asd_list)    # ASD per class
    metrics_list.extend(ahd_list)    # AHD per class

    return metrics_list
def predict_one_case(case_name, args, models, device, output_dir):
    clean_name = case_name.replace("-T1", "")

    t1_path = os.path.join(args.input_folder, "x_t1_data", f"{clean_name}-T1.nii.gz")
    dec_path = os.path.join(args.input_folder, "x_dec_data", f"{clean_name}-DEC.nii.gz")

    t1_img    = nib.load(t1_path)
    dec_img    = nib.load(dec_path)

    t1_data   = t1_img.get_fdata()
    dec_data   = dec_img.get_fdata()

    affine    = t1_img.affine
    orig_shape= t1_img.shape
    spacing   = nib.affines.voxel_sizes(affine)

    # Crop
    (h0,h1),(w0,w1),(d0,d1) = get_auto_crop_coords(orig_shape)
    t1_crop   = t1_data[h0:h1, w0:w1, d0:d1]
    dec_crop   = dec_data[h0:h1, w0:w1, d0:d1]

    dataset = CN_InferDataset(t1_crop, dec_crop)
    loader  = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)

    num_classes = 5
    avg_logits = np.zeros((num_classes, 128, 160, 128), dtype=np.float32)

    with torch.no_grad():
        for t1_batch, dec_batch, z_batch in loader:
            t1_batch = t1_batch.to(device)
            dec_batch = dec_batch.to(device)

            pred_sum = np.zeros((t1_batch.shape[0], num_classes, 128, 160), dtype=np.float32)
            for model in models:
                if args.model == "CNTSeg_V1_T1DEC":
                    fold_t1, fold_dec, fold_fusion = model
                    feat1 = fold_t1(t1_batch)
                    feat2 = fold_dec(dec_batch)
                    out = fold_fusion(feat1, feat2)
                elif args.model == "CNTSegV2_Dedicated":
                    out, _ = model(t1_batch, dec_batch)
                else:
                    out = model(t1_batch, dec_batch)
                pred_sum += out.cpu().numpy()

            pred_avg = pred_sum / len(models)
            for i, z in enumerate(z_batch):
                avg_logits[:, :, :, z] = pred_avg[i]

    pred_crop = np.argmax(avg_logits, axis=0).astype(np.uint8)
    pred_full = np.zeros(orig_shape, dtype=np.uint8)
    pred_full[h0:h1, w0:w1, d0:d1] = pred_crop

    out_path = os.path.join(output_dir, f"{clean_name}-ensemble.nii.gz")
    nib.save(nib.Nifti1Image(pred_full, affine), out_path)
    print(f"✅ Saved: {out_path}")

    metrics = None
    if args.gt_folder:
        gt_path = os.path.join(args.gt_folder, f"{clean_name}-label.nii.gz")
        if os.path.exists(gt_path):
            gt = nib.load(gt_path).get_fdata()
            metrics = compute_all_metrics(pred_full, gt, spacing)
    return clean_name, metrics

def main():
    args = get_parser()
    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    RESULT_ROOT= args.result_root
    output_dir = os.path.join(RESULT_ROOT, args.model, "Ensemble_Results")
    os.makedirs(output_dir, exist_ok=True)

    # 加载 5 折模型
    folds = [f"fold_{i}" for i in range(5)]
    models = []
    global model_t1, model_dec

    for fold in folds:
        ##################################################################################
        if args.model == "PHMNet":
            model = PHMNet().to(device)
        ####################################################################################
        elif args.model == "CNTSeg_V1_T1DEC":
            t1_model_path = os.path.join(args.result_root, "CNTSeg_V1_T1DEC", "UnetT1", fold, "BEST_MODEL.pth")
            dec_model_path = os.path.join(args.result_root, "CNTSeg_V1_T1DEC", "UnetDEC", fold, "BEST_MODEL.pth")
            model_t1 = cntsegunet(1, 5).to(device)
            model_dec = cntsegunet(3, 5).to(device)
            model_t1.load_state_dict(torch.load(t1_model_path, map_location=device))
            model_dec.load_state_dict(torch.load(dec_model_path, map_location=device))
            model_t1.eval()
            model_dec.eval()
            for param in model_t1.parameters(): param.requires_grad = False
            for param in model_dec.parameters(): param.requires_grad = False
            model = cntsegfusionnet_2(10, 5).to(device)
        ####################################################################################
        elif args.model == "CNTSegV2_Dedicated":
            model = CNTSegV2_Dedicated(1, 3, 5).to(device)
        ####################################################################################
        model_path = os.path.join(RESULT_ROOT, args.model, fold, "BEST_MODEL.pth")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        if args.model == "CNTSeg_V1_T1DEC":
            models.append(
                (model_t1, model_dec, model)
            )
        else:
            models.append(model)

    all_metrics_data = []
    summary_path = os.path.join(output_dir, "ensemble_metrics_summary.txt")

    if args.mode == "batch":

        example_dir = os.path.join(args.input_folder, "x_t1_data")
        files = sorted([f[:-7] for f in os.listdir(example_dir) if f.endswith(".nii.gz")])
        for fname in tqdm(files, desc="Predicting"):
            case_name, metrics = predict_one_case(fname, args, models, device, output_dir)
            if metrics:
                all_metrics_data.append((case_name, metrics))

    if all_metrics_data:
        with open(summary_path, "w") as f:
            header = (
                "Case\tDice_avg\tJac_avg\tASD_avg\tAHD_avg\t"
                "Dice_ON\tDice_OCN\tDice_TGN\tDice_FVN\t"
                "Jac_ON\tJac_OCN\tJac_TGN\tJac_FVN\t"
                "ASD_ON\tASD_OCN\tASD_TGN\tASD_FVN\t"
                "AHD_ON\tAHD_OCN\tAHD_TGN\tAHD_FVN"
            )
            f.write("="*220+"\n")
            f.write(header+"\n")
            f.write("="*220+"\n")
            metrics_array = []
            for case_name, metrics in all_metrics_data:
                row = f"{case_name}\t"+"\t".join([f"{v:.6f}" for v in metrics])
                f.write(row+"\n")
                metrics_array.append(metrics)

            if len(metrics_array)>1:
                arr = np.array(metrics_array)
                m = np.mean(arr, axis=0)
                s = np.std(arr, axis=0)
                ms = [f"{a:.4f}±{b:.4f}" for a,b in zip(m,s)]
                f.write("="*220+"\n")
                f.write("MEAN±STD\t"+"\t".join(ms)+"\n")
                f.write("="*220+"\n")
        print(f"\n🎉 Summary saved: {summary_path}")

        arr = np.array([m for _,m in all_metrics_data])
        m = np.mean(arr, axis=0)
        names = [
            "Dice_avg","Jac_avg","ASD_avg","AHD_avg",
            "Dice_ON","Dice_OCN","Dice_TGN","Dice_FVN",
            "Jac_ON","Jac_OCN","Jac_TGN","Jac_FVN",
            "ASD_ON","ASD_OCN","ASD_TGN","ASD_FVN",
            "AHD_ON","AHD_OCN","AHD_TGN","AHD_FVN"
        ]
        print("\n" + "="*90)
        print(" FINAL MEAN METRICS ".center(90, " "))
        print("="*90)
        for i in range(0, len(names), 5):
            line = ""
            for j in range(5):
                if i+j < len(names):
                    line += f"{names[i+j]}: {m[i+j]:.4f}   "
            print(line)
        print("="*90)

    print("\n🎉 ALL DONE!")

if __name__ == "__main__":
    main()
