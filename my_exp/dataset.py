import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image

class MSDDataset(Dataset):
    def __init__(self, root_dir, task_name, img_size=128, modality="CT", subset_size=None):
        
        path_standard = os.path.join(root_dir, task_name)
        # Check Nested Path:   ./Task09_Spleen/Task09_Spleen/imagesTr
        path_nested = os.path.join(root_dir, task_name, task_name)
        
        if os.path.exists(os.path.join(path_standard, "imagesTr")):
            self.task_path = path_standard
        elif os.path.exists(os.path.join(path_nested, "imagesTr")):
            self.task_path = path_nested
        else:
            raise ValueError(f"CRITICAL ERROR: Could not find 'imagesTr' folder.\nChecked locations:\n1. {path_standard}\n2. {path_nested}")

        self.img_dir = os.path.join(self.task_path, "imagesTr")
        self.lbl_dir = os.path.join(self.task_path, "labelsTr")
        self.modality = modality
        self.img_size = img_size
        self.slices = []
        
        img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.nii.gz")))
        lbl_files = sorted(glob.glob(os.path.join(self.lbl_dir, "*.nii.gz")))
        
        if len(img_files) == 0:
            raise ValueError(f"No data found in {self.img_dir}. Check paths!")
            
        print(f"Scanning {len(img_files)} volumes for Task: {task_name}...")
        print(f"  -> Source: {self.img_dir}")
        
        count = 0
        for img_p, lbl_p in zip(img_files, lbl_files):
            if subset_size and count >= subset_size: break
            try:
                nii_lbl = nib.load(lbl_p)
                data_lbl = nib.as_closest_canonical(nii_lbl).get_fdata()
                #nii_lbl contains the metadata
                for i in range(data_lbl.shape[2]):
                    if subset_size and count >= subset_size: break
                    if np.max(data_lbl[:,:,i]) > 0: 
                        self.slices.append((img_p, lbl_p, i))
                        count += 1
            except Exception as e:
                print(f"Skipping corrupt file {img_p}: {e}")
                
        print(f"  -> Valid Slices Found: {len(self.slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_p, lbl_p, s_idx = self.slices[idx]
        
        img = nib.as_closest_canonical(nib.load(img_p)).dataobj[..., s_idx]
        lbl = nib.as_closest_canonical(nib.load(lbl_p)).dataobj[..., s_idx]
        # img and lbl are arrays
        if self.modality == "CT":
            img = np.clip(img, -150, 250)
            img = (img - (-150)) / (250 - (-150))
        else:
            p1 = np.percentile(img, 1)
            p99 = np.percentile(img, 99)
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1 + 1e-8)

        img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize((self.img_size, self.img_size), Image.BILINEAR)
        lbl_pil = Image.fromarray(((lbl > 0) * 255).astype(np.uint8)).resize((self.img_size, self.img_size), Image.NEAREST)
        # nearest because val of each pixel of label can 0/1 not 0.5 or something else
        
        img_np = np.array(img_pil) / 255.0
        lbl_np = np.array(lbl_pil) / 255.0
        
        return torch.from_numpy(img_np).float().unsqueeze(0), torch.from_numpy(lbl_np).float().unsqueeze(0)