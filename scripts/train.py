# train.py

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from dataloading.datasets import LadosDataset, SyntheticDataset

# ============================================================================
# Augmentation: Physics-Aware Diffraction Erasing
# ============================================================================
class DiffractionErasing:
    """Physically meaningful masking for diffraction patterns"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img

        erase_type = torch.rand(1)
        if erase_type < 0.3:
            return self._beamstop_mask(img)
        elif erase_type < 0.6:
            return self._dead_module_mask(img)
        else:
            return self._wedge_mask(img)

    def _beamstop_mask(self, img):
        h, w = img.shape[-2:]
        center = (h // 2, w // 2)
        radius = int(torch.rand(1) * 0.1 * min(h, w) + 0.02 * min(h, w))
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        mask = ((x - center[1])**2 + (y - center[0])**2) <= radius**2
        img_masked = img.clone()
        img_masked[..., mask] = 0
        return img_masked

    def _dead_module_mask(self, img):
        h, w = img.shape[-2:]
        rect_h = int(torch.rand(1) * 0.15 * h + 0.05 * h)
        rect_w = int(torch.rand(1) * 0.15 * w + 0.05 * w)
        top = torch.randint(0, h - rect_h, (1,)).item()
        left = torch.randint(0, w - rect_w, (1,)).item()
        img_masked = img.clone()
        img_masked[..., top:top+rect_h, left:left+rect_w] = 0
        return img_masked

    def _wedge_mask(self, img):
        h, w = img.shape[-2:]
        center = (h // 2, w // 2)
        start_angle = torch.rand(1) * 360
        wedge_size = torch.rand(1) * 60 + 15
        end_angle = start_angle + wedge_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        angles = torch.atan2(y - center[0], x - center[1]) * 180 / np.pi
        angles = (angles + 360) % 360
        if end_angle <= 360:
            mask = (angles >= start_angle) & (angles <= end_angle)
        else:
            mask = (angles >= start_angle) | (angles <= (end_angle % 360))
        img_masked = img.clone()
        img_masked[..., mask] = 0
        return img_masked

# ============================================================================
# Data Module
# ============================================================================
class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, synthetic_dir: str, train_val_split: float = 0.8, batch_size: int = 1, num_workers: int = 4):
        super().__init__()
        self.synthetic_dir = synthetic_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = T.Compose([
            T.RandomRotation(180),
            T.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.9, 1.5)),
            DiffractionErasing(p=1.0),
        ])

    def setup(self, stage=None):
        # Uses the refactored SyntheticDataset (expects root_dir)
        full_dataset = SyntheticDataset(root_dir=self.synthetic_dir)
        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        self.train_dataset = DatasetWithTransform(train_dataset, self.train_transforms)
        self.val_dataset = DatasetWithTransform(val_dataset, None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)

class DatasetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        image, labels = self.subset[idx]
        if self.transform: image = self.transform(image)
        return image, labels

# ============================================================================
# Training Logic
# ============================================================================
def train_model(args):
    """
    Main training function configured via argparse
    """
    
    # --- 1. Setup Data ---
    data_module = SyntheticDataModule(
        synthetic_dir=args.synthetic_dir,
        train_val_split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # --- 2. Setup Model ---
    from models import MIT_B2  
    
    model = MIT_B2(
        encoder_name="mit_b2",
        num_phases=3,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs
    )

    # --- 3. Setup Logging & Checkpoints ---
    logger = TensorBoardLogger(save_dir=args.log_dir, name="MIT-B2-Run")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val_loss",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )

    # --- 4. Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=20)],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        precision=16, # Mixed precision
        accumulate_grad_batches=args.accumulate_grad
    )

    print(f"Starting training on data: {args.synthetic_dir}")
    trainer.fit(model, data_module)

    # --- 5. Evaluation (Optional) ---
    if args.evaluate_lados and args.lados_dir:
        print("Evaluating on real Lados dataset...")
        best_model = MIT_B2.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model.eval()
        
        # Uses the refactored LadosDataset (expects root_dir)
        lados_dataset = LadosDataset(root_dir=args.lados_dir)
        lados_loader = DataLoader(lados_dataset, batch_size=args.batch_size, shuffle=False)
        
        def evaluate_model_on_real_data(model, dataloader, device='cuda', normalize_predictions=True):
            """
            Evaluates a model on data providing only (image, target) pairs.
            """
            model.eval()
            model = model.to(device)

            all_predictions = []
            all_targets = []
            all_normalized_predictions = []

            loss_fn = nn.MSELoss()
            total_loss, total_mae, num_batches = 0, 0, 0

            with torch.no_grad():
                for batch in dataloader:
                    images, targets = batch
                    images, targets = images.to(device), targets.to(device)

                    predictions = model(images)

                    # Metric calculation logic
                    if normalize_predictions:
                        alpha_beta_sum = torch.clamp(predictions[:, 0] + predictions[:, 1], min=1e-8)
                        normalized_preds = predictions.clone()
                        normalized_preds[:, 0] = predictions[:, 0] / alpha_beta_sum
                        normalized_preds[:, 1] = predictions[:, 1] / alpha_beta_sum
                        loss = loss_fn(normalized_preds[:, :2], targets[:, :2])
                        mae = torch.mean(torch.abs(normalized_preds[:, :2] - targets[:, :2]))
                        all_normalized_predictions.append(normalized_preds.cpu().numpy())
                    else:
                        loss = loss_fn(predictions, targets)
                        mae = torch.mean(torch.abs(predictions - targets))

                    total_loss += loss.item()
                    total_mae += mae.item()
                    num_batches += 1

                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            avg_loss = total_loss / num_batches
            avg_mae = total_mae / num_batches

            result = {
                'loss': avg_loss,
                'mae': avg_mae,
                'predictions': np.concatenate(all_predictions),
                'targets': np.concatenate(all_targets),
            }
            if normalize_predictions:
                result['normalized_predictions'] = np.concatenate(all_normalized_predictions)

            return result

# ============================================================================
# Command Line Interface
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIT-B2 on Synthetic Diffraction Data")

    # Path Arguments (The most important part for reproducibility)
    parser.add_argument('--synthetic_dir', type=str, required=True, help="Path to synthetic training data (.npy files)")
    parser.add_argument('--lados_dir', type=str, default=None, help="Path to real validation data (Lados TIFs)")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints", help="Directory to save models")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory for TensorBoard logs")

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1.25e-6)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--accumulate_grad', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--evaluate_lados', action='store_true', help="Run evaluation on real data after training")

    args = parser.parse_args()

    # Create dirs if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_model(args)