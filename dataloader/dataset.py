import torch 
import numpy as np
from einops import repeat
from utils.utils import compute_patch_mask


class PatchedDataset(torch.utils.data.Dataset):
    def __init__(self, img, split_gt, leakage_gt, patch_size, use_pca=False, pcs=None, unlabeled_id=-1, shuffle=True,
        padding_mode='reflect', padding_value=0, stride=1, return_all=False):
        """
        leakage_gt: is gt of others splits 
        """
        super().__init__()
        self.split_gt = torch.from_numpy(split_gt)
        if leakage_gt is not None and patch_size >= 3:
            self.leakage_gt = torch.from_numpy(leakage_gt)
        else:
            self.leakage_gt = None
        self.img = img
        self.stride = stride
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.pad_mode = padding_mode
        self.pad_value = padding_value
        self.unlabeled_id = unlabeled_id
        self.use_pca = use_pca
        self.pcs = pcs
        self.retrun_all=return_all
        
        self.gt_positions = self.compute_positions(self.split_gt, unlabeled_id)
        if self.shuffle:
            np.random.default_rng().shuffle(self.gt_positions)
            
        if self.leakage_gt is not None:
            self.leakage_positions = self.compute_positions(self.leakage_gt, unlabeled_id)
        
        if self.use_pca:
            B, H, W = img.shape
            n_pcs = pcs.n_components_
            img = img.reshape(B, -1).transpose(1, 0)
            img = pcs.transform(img).transpose(1, 0)
            img = img.reshape(n_pcs, H, W)
        self.img_padded = self.padded_img(torch.from_numpy(img), self.pad_size, self.pad_mode, self.pad_value)
 
        
    def compute_positions(self, gt, unlabeled_id):
        rows, cols = torch.where(gt != unlabeled_id)
        positions = list(zip(rows.tolist(), cols.tolist()))
        if self.stride > 1:
            R, C = gt.shape
            positions = [
                (r, c)
                for r in range(0, R, self.stride)
                for c in range(0, C, self.stride)
                if (r, c) in positions
            ]
        return positions
            
    def padded_img(self, img, pad_size, pad_mode, pad_value):
        if pad_mode == "constant":
            return torch.nn.functional.pad(
                img,
                pad=(pad_size, pad_size, pad_size, pad_size),
                mode=pad_mode,
                value=pad_value)
        elif pad_mode == "reflect":
            return torch.nn.functional.pad(
                img,
                pad=(pad_size, pad_size, pad_size, pad_size),
                mode=pad_mode)
        else:
            raise ValueError("UNK paddind mode.")
            


    def __len__(self):
        return len(self.gt_positions)

    def __getitem__(self, idx):
        row, col = self.gt_positions[idx]
        # Adjust for padding
        row_p, col_p = row + self.pad_size, col + self.pad_size
        patch = self.img_padded[
            :,
            row_p - self.pad_size : row_p + self.pad_size + 1,
            col_p - self.pad_size : col_p + self.pad_size + 1
        ]
        label = self.split_gt[row, col]
        
        if self.leakage_gt is not None:
            mask = compute_patch_mask(row, col, self.patch_size, self.leakage_positions)
            mask = repeat(mask, 'h w -> c h w', c=patch.shape[0])
            mask = torch.from_numpy(mask).type(torch.float32)
            patch *= mask
            
        else: 
            mask = -1  # dummy value if no leakage gt
        
        if self.retrun_all:
            return patch.type(torch.float32), label.long(), mask, (row, col)
        else:
            return patch.type(torch.float32), label.long()
