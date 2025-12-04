import os
import argparse
import csv
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.regression import R2Score, MeanAbsoluteError, MeanSquaredError

# --- Local Imports ---
from dataloading.datasets import LadosDataset, StrantzaDataset
from models.mit_b2 import MIT_B2

def get_dataset(dataset_type, data_dir):
    if dataset_type.lower() == 'lados':
        return LadosDataset(root_dir=data_dir, normalize_percentile=True)
    elif dataset_type.lower() == 'strantza':
        return StrantzaDataset(root_dir=data_dir, normalize_percentile=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def run_finetuning_experiment(args):
    pl.seed_everything(args.seed)
    
    # --- 1. Load Data ---
    print(f"Loading {args.dataset_type} dataset from {args.data_dir}...")
    
    # Check if directory exists first
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist.")
        return

    full_dataset = get_dataset(args.dataset_type, args.data_dir)
    total_samples = len(full_dataset)
    print(f"Total samples found: {total_samples}")

    if total_samples == 0:
        print("Error: Dataset is empty. Check if labels.json exists in the folder.")
        return

    # Initialize results logging
    results_file = os.path.join(args.output_dir, f"results_{args.dataset_type}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train_size', 'test_size', 'mae', 'mse', 'r2', 'checkpoint_path'])

    # --- 2. Progressive Fine-Tuning Loop ---
    for k_shots in args.train_sizes:
        # SKIP LOGIC: If we don't have enough data for k_shots + at least 1 test item
        if k_shots >= total_samples:
            print(f"Skipping train_size={k_shots} (Dataset only has {total_samples} samples)")
            continue

        print(f"\n{'-'*60}")
        print(f"Running Experiment: Train Size = {k_shots}")
        print(f"{'-'*60}")

        # --- Data Splitting ---
        test_size = total_samples - k_shots
        
        # Ensure we have at least 1 test sample
        if test_size < 1:
            print(f"Skipping: Not enough data left for testing.")
            continue

        train_subset, test_subset = random_split(
            full_dataset, 
            [k_shots, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # --- Load Model ---
        # If a pretrained checkpoint is provided (e.g., from Synthetic), load weights.
        if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
            print(f"Loading weights from: {args.pretrained_ckpt}")
            model = MIT_B2.load_from_checkpoint(
                args.pretrained_ckpt,
                strict=False, 
                learning_rate=args.lr,
                max_epochs=args.epochs,
                warmup_epochs=0 
            )
        else:
            print("Warning: No pretrained checkpoint found. Training from scratch.")
            model = MIT_B2(
                encoder_name="mit_b2", 
                num_phases=3,
                learning_rate=args.lr,
                max_epochs=args.epochs
            )

        # Optional: Freeze Encoder
        if args.freeze_encoder:
            print("Freezing encoder layers...")
            for param in model.encoder.parameters():
                param.requires_grad = False

        # --- Setup Trainer ---
        run_name = f"ft_{args.dataset_type}_k{k_shots}"
        logger = TensorBoardLogger(save_dir=args.log_dir, name=run_name)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, run_name),
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=1
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=20)],
            log_every_n_steps=1, # Important for small datasets to see logs frequently
            check_val_every_n_epoch=5 # Don't validate every single epoch if dataset is tiny
        )

        # --- Train ---
        trainer.fit(model, train_loader, val_dataloaders=test_loader)

        # --- Evaluation ---
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            best_model = MIT_B2.load_from_checkpoint(best_model_path)
            best_model.eval()
            best_model.to(model.device)
            
            mae = MeanAbsoluteError().to(model.device)
            mse = MeanSquaredError().to(model.device)
            r2 = R2Score().to(model.device)
            
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    imgs, targets = batch
                    imgs = imgs.to(model.device)
                    targets = targets.to(model.device)
                    
                    preds = best_model(imgs)
                    
                    all_preds.append(preds)
                    all_targets.append(targets)
            
            if len(all_preds) > 0:
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)

                final_mae = mae(all_preds, all_targets).item()
                final_mse = mse(all_preds, all_targets).item()
                final_r2 = r2(all_preds, all_targets).item()

                print(f"Results for k={k_shots}: MAE={final_mae:.4f}, R2={final_r2:.4f}")

                # Append to CSV
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([k_shots, test_size, final_mae, final_mse, final_r2, best_model_path])
            else:
                print("Warning: No predictions generated (Empty test set?).")
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Fine-Tuning for Diffraction Analysis")

    # Dataset Args
    parser.add_argument('--dataset_type', type=str, required=True, choices=['lados', 'strantza'], help="Which dataset to use")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing images and labels.json")
    
    # Model Args
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help="Path to the synthetic pre-trained checkpoint")
    parser.add_argument('--freeze_encoder', action='store_true', help="Freeze encoder weights during fine-tuning")
    
    # Experiment Args
    # Default sizes are standard, but the script will auto-skip ones too big for the dataset
    parser.add_argument('--train_sizes', nargs='+', type=int, default=[1, 2, 3, 5, 10, 20, 50, 100], help="List of training set sizes to loop through")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2) # Reduced default for safer demo runs
    
    # Output
    parser.add_argument('--output_dir', type=str, default="./output/results", help="Where to save results CSV")
    parser.add_argument('--checkpoint_dir', type=str, default="./output/checkpoints_ft", help="Where to save fine-tuned models")
    parser.add_argument('--log_dir', type=str, default="./output/logs_ft", help="TensorBoard logs")

    args = parser.parse_args()
    
    run_finetuning_experiment(args)