import numpy as np
import os
import nibabel as nib
from torch.utils.data import Dataset

class CN_MyTrainDataset(Dataset):
    def __init__(self, x_t1_dir, x_dec_dir, y_label_dir, x_transform=None, y_transform=None):
        self.names = sorted([f.replace(".nii.gz", "") for f in os.listdir(x_t1_dir) if f.endswith(".nii.gz")])
        self.x_t1_dir = x_t1_dir
        self.x_dec_dir = x_dec_dir
        self.y_label_dir = y_label_dir
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        name = self.names[index]

        x1 = nib.load(os.path.join(self.x_t1_dir, name + ".nii.gz")).get_fdata().astype(np.float32)
        x2 = nib.load(os.path.join(self.x_dec_dir, name + ".nii.gz")).get_fdata().astype(np.float32)
        lab = nib.load(os.path.join(self.y_label_dir, name + ".nii.gz")).get_fdata()

        x1 = np.nan_to_num(x1)
        x2 = np.nan_to_num(x2)
        lab = np.nan_to_num(lab)

        lab = lab.squeeze().astype(np.int64)   # 确保整数标签
        y_bg  = (lab == 0).astype(np.float32)
        y_on  = (lab == 1).astype(np.float32)
        y_ocn = (lab == 2).astype(np.float32)
        y_tgn = (lab == 3).astype(np.float32)
        y_fvn = (lab == 4).astype(np.float32)

        if self.x_transform is not None:
            x1 = self.x_transform(x1)
            x2 = self.x_transform(x2)
        if self.y_transform is not None:
            y_bg  = self.y_transform(y_bg)
            y_on  = self.y_transform(y_on)
            y_ocn = self.y_transform(y_ocn)
            y_tgn = self.y_transform(y_tgn)
            y_fvn = self.y_transform(y_fvn)

        return x1, x2, y_bg, y_on, y_ocn, y_tgn, y_fvn, name

    def __len__(self):
        return len(self.names)