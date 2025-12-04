# datasets.py (Refactored for Demo)
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json
import re

# --- Dataset 1: Bonnie et al. ---
class LadosDataset(Dataset):
    """
    Loads TIF images from a directory and labels from a labels.json file.
    Expects structure: 
       root_dir/
          image1.tif
          image2.tif
          labels.json  --> {"image1.tif": [a, b, g], ...}
    """
    def __init__(self, root_dir, normalize_percentile=False, p_low=1.0, p_high=99.9):
        self.root_dir = root_dir
        self.normalize_percentile = normalize_percentile
        self.p_low = p_low
        self.p_high = p_high
        
        # Load Metadata
        json_path = os.path.join(root_dir, 'labels.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"labels.json not found in {root_dir}. Run create_demo_data.py first.")
            
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.image_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.root_dir, filename)
        
        # Load Image
        image = np.array(Image.open(image_path)).astype(np.float32)
        if image.ndim == 2:
            image = image[np.newaxis, ...] # Add channel dim

        # Normalize
        if self.normalize_percentile:
            p_low_val, p_high_val = np.percentile(image, [self.p_low, self.p_high])
            denominator = p_high_val - p_low_val
            if denominator > 1e-8:
                image = np.clip(image, p_low_val, p_high_val)
                image = (image - p_low_val) / denominator
            else:
                image = np.zeros_like(image, dtype=np.float32)
        else:
            # Original Normalization factor
            image = image / 14582.3

        # Load Label
        label = self.metadata[filename]
        
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.float32)


# --- Dataset 2: Strantza et al. ---
class StrantzaDataset(Dataset):
    """
    Loads NPY images from a directory and labels from a labels.json file.
    """
    def __init__(self, root_dir, normalize_percentile=True, p_low=1.0, p_high=99.9):
        self.root_dir = root_dir
        self.normalize_percentile = normalize_percentile
        self.p_low = p_low
        self.p_high = p_high

        # Load Metadata
        json_path = os.path.join(root_dir, 'labels.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"labels.json not found in {root_dir}. Run create_demo_data.py first.")
            
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.image_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        file_path = os.path.join(self.root_dir, filename)
        
        # Load Image
        img = np.load(file_path).astype(np.float32)
        
        # Normalize
        if self.normalize_percentile:
            p_low_val = np.percentile(img, self.p_low)
            p_high_val = np.percentile(img, self.p_high)
            img = np.clip(img, p_low_val, p_high_val)
            denominator = p_high_val - p_low_val
            if denominator > 0:
                img = (img - p_low_val) / denominator
            else:
                img = img - p_low_val # Fallback if range is 0

        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        
        # Load Label
        label = self.metadata[filename]

        return img, torch.tensor(label, dtype=torch.float32)


# --- Dataset 3: Synthetic (Standard) ---
class SyntheticDataset(Dataset):
    """
    Loads NPY images, labels parsed from filenames.
    """
    def __init__(self, directory):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith(".npy")]
        self.pattern = re.compile(r"A([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)B([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)G([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
        self.GLOBAL_MAX = 17482.550781

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.directory, filename)

        match = self.pattern.search(filename)
        if not match:
            print(f"Warning: {filename} pattern mismatch. Returning zero label.")
            labels = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            alpha, beta, gamma = map(float, match.groups())
            labels = torch.tensor([alpha, beta, gamma], dtype=torch.float32)
        
        image = np.load(filepath).astype(np.float32)
        image = image / self.GLOBAL_MAX
        image = torch.from_numpy(image).unsqueeze(0)

        return image, labels