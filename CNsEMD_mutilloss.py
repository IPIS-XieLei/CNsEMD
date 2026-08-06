import torch.nn as nn
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch.nn.functional as F

def diceCoeff_single(pred, gt, num_classes, eps=1e-5):
    r""" 单标签多分类Dice系数（每个类别单独计算） """
    pred = torch.argmax(pred, dim=1)  # 直接转类别索引
    dice_list = []
    for c in range(1, num_classes):  # 跳过背景（0）
        pred_c = (pred == c).float()
        gt_c = (gt == c).float()
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        dice = (2 * intersection + eps) / (union + eps)
        dice_list.append(dice.item())
    return dice_list

class SoftDiceLossSingle(nn.Module):
    """ 单标签多分类Dice Loss """
    __name__ = 'dice_loss_single'
    def __init__(self, num_classes, eps=1e-5):
        super(SoftDiceLossSingle, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, y_pred, y_true):
        # y_pred: (B, C, H, W) logits
        # y_true: (B, H, W) 类别索引
        y_pred = torch.softmax(y_pred, dim=1)
        loss = 0
        for c in range(1, self.num_classes):  # 跳过背景
            pred_c = y_pred[:, c, :, :]
            gt_c = (y_true == c).float()
            intersection = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice = (2 * intersection + self.eps) / (union + self.eps)
            loss += (1 - dice)
        return loss / (self.num_classes - 1)  # 平均到每个类别

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    assert len(img_gt.shape) == len(out_shape) == 4
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)
    for b in range(out_shape[0]):  # batch size
        for c in range(out_shape[1]):  # channel
            posmask = img_gt[b][c].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis) + 1e-7) - (
                            posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis) + 1e-7)
                # sdf = (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)) - (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf
    return normalized_sdf


def mmd_direction_loss(pred_dir, target_dir, mask, sigma=1.0):
    """
    Maximum Mean Discrepancy (MMD) loss for direction vectors.
    Args:
        pred_dir: [B,3,H,W] predicted direction vectors (normalized)
        target_dir: [B,3,H,W] target direction vectors (not normalized, will be normalized inside)
        mask: [B,1,H,W] binary mask (1 for foreground, 0 for background)
        sigma: Gaussian kernel bandwidth
    Returns:
        scalar loss
    """
    # 归一化 target_dir
    target_dir = F.normalize(target_dir, dim=1, eps=1e-8)

    B, C, H, W = pred_dir.shape
    pred_flat = pred_dir.view(B, C, -1)  # [B,3,N]
    target_flat = target_dir.view(B, C, -1)  # [B,3,N]
    mask_flat = mask.view(B, 1, -1)  # [B,1,N]

    loss = 0.0
    for b in range(B):
        idx = mask_flat[b, 0].bool()  # foreground indices
        if idx.sum() < 2:  # at least 2 points needed for kernel
            continue
        X = pred_flat[b, :, idx].T  # [N_fg, 3]
        Y = target_flat[b, :, idx].T  # [N_fg, 3]

        # Gaussian kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        # Compute pairwise squared Euclidean distances
        # Use pdist2 style
        # For X and Y, we need K_XX, K_YY, K_XY
        def kernel_matrix(A, B):
            # A: [N, D], B: [M, D] -> K: [N, M]
            sqdist = torch.cdist(A, B) ** 2  # [N, M]
            return torch.exp(-sqdist / (2 * sigma ** 2))

        K_XX = kernel_matrix(X, X)  # [N_fg, N_fg]
        K_YY = kernel_matrix(Y, Y)
        K_XY = kernel_matrix(X, Y)

        n = X.shape[0]
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        loss += mmd
    return loss / B
