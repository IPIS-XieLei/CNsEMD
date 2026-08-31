import json
import time
import os
import numpy as np
import nibabel as nib
import argparse
import shutil

def get_parser():
    parser = argparse.ArgumentParser(description="Data Preprocessing for CNsEMD")
    parser.add_argument('--data_root', type=str, default="/path/to/CNsEMD")
    parser.add_argument('--split_json', type=str,
                        default="/path/to/CNsEMD/splits_final_train.json")
    parser.add_argument('--test_split_json', type=str, default="/path/to/CNsEMD/splits_final_test.json")
    parser.add_argument('--output_root', type=str, default="/path/to/CNsEMD_preprocessed_data")
    parser.add_argument('--modality_type', type=str, default="five", choices=["five", "two"],
                        help="'five': T1+T2+FA+DEC+Peaks+GT; 'two': T1+DEC+GT only")
    return parser.parse_args()

def load_nii(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine

def normalize_t1(arr):
    arr = np.nan_to_num(arr)
    low = np.percentile(arr, 1)
    high = np.percentile(arr, 99)
    arr = np.clip(arr, low, high)
    mean = arr.mean()
    std = arr.std() + 1e-8
    return ((arr - mean) / std).astype(np.float32)

def normalize_fa(arr):
    arr = np.nan_to_num(arr)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)

def normalize_t2(arr):
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

def peaks_keep_original(arr):
    arr = np.nan_to_num(arr)
    return arr.astype(np.float32)

def save_nii(data, affine, path):
    nib.save(nib.Nifti1Image(data, affine), path)

def make_folders(base, modality_type="five"):
    modalities = ["x_t1_data", "x_dec_data", "y_label_data"]
    if modality_type == "five":
        modalities += ["x_t2_data", "x_fa_data", "x_peaks_data"]
    for mod in modalities:
        os.makedirs(os.path.join(base, mod), exist_ok=True)

def get_auto_crop(vol_shape):
    target = (128, 160, 128)
    slices = []
    for dim in range(3):
        raw = vol_shape[dim]
        tgt = target[dim]
        start = (raw - tgt) // 2
        end = start + tgt
        slices.append(slice(start, end))
    return tuple(slices)

def process_one_case(prefix_in_json, t1_dir, t2_dir, fa_dir, dec_dir, pk_dir, gt_dir, save_dir, modality_type="five", augment=False):
    file_prefix = prefix_in_json.replace("CNs_", "")

    t1_path = os.path.join(t1_dir, f"{file_prefix}-T1.nii.gz")
    dec_path = os.path.join(dec_dir, f"{file_prefix}-DEC.nii.gz")
    gt_path = os.path.join(gt_dir, f"{file_prefix}-label.nii.gz")

    t1, aff = load_nii(t1_path)
    dec, _ = load_nii(dec_path)
    lab, _ = load_nii(gt_path)

    t1 = np.squeeze(t1)
    dec = np.squeeze(dec)
    lab = np.squeeze(lab)

    if modality_type == "five":
        t2_path = os.path.join(t2_dir, f"{file_prefix}-T2.nii.gz")
        fa_path = os.path.join(fa_dir, f"{file_prefix}-FA.nii.gz")
        pk_path = os.path.join(pk_dir, f"{file_prefix}-Peaks.nii.gz")
        t2, _ = load_nii(t2_path)
        fa, _ = load_nii(fa_path)
        pk, _ = load_nii(pk_path)
        t2 = np.squeeze(t2)
        fa = np.squeeze(fa)
        pk = np.squeeze(pk)

    crop = get_auto_crop(t1.shape)
    t1 = t1[crop]
    dec = dec[crop[0], crop[1], crop[2], :]
    lab = lab[crop]
    if modality_type == "five":
        t2 = t2[crop]
        fa = fa[crop]
        pk = pk[crop[0], crop[1], crop[2], :]

    saved = 0
    for z in range(t1.shape[2]):
        if np.count_nonzero(lab[..., z]) == 0:
            continue

        t1_z = normalize_t1(t1[..., z])
        dec_z = dec_keep_original(dec[..., z, :])
        lab_z = lab[..., z].astype(np.uint8)

        slice_name = f"{file_prefix}_z{z:03d}"
        save_nii(t1_z, aff, os.path.join(save_dir, "x_t1_data", f"{slice_name}.nii.gz"))
        save_nii(dec_z, aff, os.path.join(save_dir, "x_dec_data", f"{slice_name}.nii.gz"))
        save_nii(lab_z, aff, os.path.join(save_dir, "y_label_data", f"{slice_name}.nii.gz"))

        if modality_type == "five":
            t2_z = normalize_t2(t2[..., z])
            fa_z = normalize_fa(fa[..., z])
            pk_z = peaks_keep_original(pk[..., z, :])
            save_nii(t2_z, aff, os.path.join(save_dir, "x_t2_data", f"{slice_name}.nii.gz"))
            save_nii(fa_z, aff, os.path.join(save_dir, "x_fa_data", f"{slice_name}.nii.gz"))
            save_nii(pk_z, aff, os.path.join(save_dir, "x_peaks_data", f"{slice_name}.nii.gz"))

        saved += 1
        # if augment:
        #     slice_aug = f"{slice_name}_aug"
        #     t1_aug = np.flip(t1_z, axis=0).copy()
        #     dec_aug = np.flip(dec_z, axis=0).copy()
        #     lab_aug = np.flip(lab_z, axis=0).copy()
        #     dec_aug[..., 0] *= -1.0
        #     save_nii(t1_aug, aff, os.path.join(save_dir, "x_t1_data", f"{slice_aug}.nii.gz"))
        #     save_nii(dec_aug, aff, os.path.join(save_dir, "x_dec_data", f"{slice_aug}.nii.gz"))
        #     save_nii(lab_aug, aff, os.path.join(save_dir, "y_label_data", f"{slice_aug}.nii.gz"))
        #     if modality_type == "five":
        #         t2_aug = np.flip(t2_z, axis=0).copy()
        #         fa_aug = np.flip(fa_z, axis=0).copy()
        #         pk_aug = np.flip(pk_z, axis=0).copy()
        #         pk_aug[..., 0::3] *= -1.0
        #         save_nii(t2_aug, aff, os.path.join(save_dir, "x_t2_data", f"{slice_aug}.nii.gz"))
        #         save_nii(fa_aug, aff, os.path.join(save_dir, "x_fa_data", f"{slice_aug}.nii.gz"))
        #         save_nii(pk_aug, aff, os.path.join(save_dir, "x_peaks_data", f"{slice_aug}.nii.gz"))
        #     saved += 1

    print(f"    Saved: {saved} slices")

def process_fold(fold_idx, train_list, val_list, args):
    print(f"\n====== Processing Fold {fold_idx} ======")
    fold_root = os.path.join(args.output_root, f"fold_{fold_idx}")
    train_save = os.path.join(fold_root, "train")
    val_save = os.path.join(fold_root, "val")
    make_folders(train_save, args.modality_type)
    make_folders(val_save, args.modality_type)

    t1_dir = os.path.join(args.data_root, "T1")
    t2_dir = os.path.join(args.data_root, "T2")
    fa_dir = os.path.join(args.data_root, "FA")
    dec_dir = os.path.join(args.data_root, "DEC")
    pk_dir = os.path.join(args.data_root, "Peaks")
    gt_dir = os.path.join(args.data_root, "GT")

    print("\n[Train Set]")
    for name in train_list:
        print(f"Train: {name}")
        process_one_case(name, t1_dir, t2_dir, fa_dir, dec_dir, pk_dir, gt_dir, train_save,
                         modality_type=args.modality_type, augment=False)

    print("\n[Val Set]")
    for name in val_list:
        print(f"Val: {name}")
        process_one_case(name, t1_dir, t2_dir, fa_dir, dec_dir, pk_dir, gt_dir, val_save,
                         modality_type=args.modality_type, augment=False)

def process_test_set(test_list, args):
    print(f"\n====== Processing Test Set ======")
    test_root = os.path.join(args.output_root, "test_set")
    make_folders(test_root, args.modality_type)

    t1_dir = os.path.join(args.data_root, "T1")
    t2_dir = os.path.join(args.data_root, "T2")
    fa_dir = os.path.join(args.data_root, "FA")
    dec_dir = os.path.join(args.data_root, "DEC")
    pk_dir = os.path.join(args.data_root, "Peaks")
    gt_dir = os.path.join(args.data_root, "GT")

    t1_save_dir = os.path.join(test_root, "x_t1_data")
    dec_save_dir = os.path.join(test_root, "x_dec_data")
    gt_save_dir = os.path.join(test_root, "y_label_data")
    if args.modality_type == "five":
        t2_save_dir = os.path.join(test_root, "x_t2_data")
        fa_save_dir = os.path.join(test_root, "x_fa_data")
        pk_save_dir = os.path.join(test_root, "x_peaks_data")

    copied_count = 0
    for name in test_list:
        print(f"Test: {name}")
        file_prefix = name.replace("CNs_", "")

        src_t1 = os.path.join(t1_dir, f"{file_prefix}-T1.nii.gz")
        src_dec = os.path.join(dec_dir, f"{file_prefix}-DEC.nii.gz")
        src_gt = os.path.join(gt_dir, f"{file_prefix}-label.nii.gz")
        dst_t1 = os.path.join(t1_save_dir, f"{file_prefix}-T1.nii.gz")
        dst_dec = os.path.join(dec_save_dir, f"{file_prefix}-DEC.nii.gz")
        dst_gt = os.path.join(gt_save_dir, f"{file_prefix}-label.nii.gz")

        shutil.copyfile(src_t1, dst_t1)
        shutil.copyfile(src_dec, dst_dec)
        shutil.copyfile(src_gt, dst_gt)

        if args.modality_type == "five":
            src_t2 = os.path.join(t2_dir, f"{file_prefix}-T2.nii.gz")
            src_fa = os.path.join(fa_dir, f"{file_prefix}-FA.nii.gz")
            src_pk = os.path.join(pk_dir, f"{file_prefix}-Peaks.nii.gz")
            dst_t2 = os.path.join(t2_save_dir, f"{file_prefix}-T2.nii.gz")
            dst_fa = os.path.join(fa_save_dir, f"{file_prefix}-FA.nii.gz")
            dst_pk = os.path.join(pk_save_dir, f"{file_prefix}-Peaks.nii.gz")
            shutil.copyfile(src_t2, dst_t2)
            shutil.copyfile(src_fa, dst_fa)
            shutil.copyfile(src_pk, dst_pk)

        copied_count += 1
        print(f"    Copied {3 if args.modality_type == 'two' else 6} files")

    print(f"\n✅ Test set done: {copied_count} cases copied to {test_root}")

if __name__ == "__main__":
    args = get_parser()

    if os.path.exists(args.output_root):
        shutil.rmtree(args.output_root)
        print("Old data removed, regenerating...")

    start = time.time()
    with open(args.split_json, "r") as f:
        splits = json.load(f)

    for i, split in enumerate(splits):
        process_fold(i, split["train"], split["val"], args)

    print("\n" + "=" * 50)
    print("Processing Independent Test Set (Direct Copy)")
    print("=" * 50)
    if os.path.exists(args.test_split_json):
        with open(args.test_split_json, "r") as f:
            test_split = json.load(f)
        process_test_set(test_split["test"], args)
    else:
        print(f"Warning: Test split json not found at {args.test_split_json}, skipped test set processing.")

    print(f"\n🎉 All done! Total time: {time.time() - start:.1f}s")
