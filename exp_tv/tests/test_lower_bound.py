"""
Lower Bound Test Script (exp_tv)
=================================
Trains U-Net on clean CT data (Phase 1), then evaluates on noisy
sparse-view FBP reconstructions (Phase 2) to get the Lower Bound.

Key design: Dataset keeps raw HU values. Physics operates on raw HU.
Normalization (CT window [-150,250] -> [0,1]) happens ONLY before U-Net
via robust_normalize(). This ensures clean and FBP images are normalized
identically.

Target: Dice >= 0.20

Usage:
    cd exp_tv
    python tests/test_lower_bound.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add parent dir to path so we can import exp_tv modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import UNet
from dataset import MSDDataset
from physics import get_physics_operator, robust_normalize


# ==========================================
#  CONFIG (mirrors main.py exactly)
# ==========================================
SEED = 42
DATA_ROOT = "../"
TASK = "Task09_Spleen"
MODALITY = "CT"
SUBSET_SIZE = 100
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
IMG_SIZE = 128
BATCH_SIZE = 4
ACCEL = 16
NOISE_SIGMA = 0.1
CENTER_FRAC = 0.08
LR_UNET = 5e-3
EPOCH_CLEAN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results_lower_bound_test")


# ==========================================
#  LOSS (same as main.py)
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs_f = inputs.view(-1)
        targets_f = targets.view(-1)
        intersection = (inputs_f * targets_f).sum()
        dice = (2. * intersection + smooth) / (inputs_f.sum() + targets_f.sum() + smooth)
        return 0.9 * bce_loss + 0.1 * (1 - dice)


# ==========================================
#  DICE METRIC
# ==========================================
def compute_dice(pred, mask):
    pred_bin = (pred > 0.5).float()
    return (2. * (pred_bin * mask).sum()) / (pred_bin.sum() + mask.sum() + 1e-8)


# ==========================================
#  MAIN TEST
# ==========================================
def main():
    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Config: ACCEL={ACCEL}, NOISE={NOISE_SIGMA}, IMG={IMG_SIZE}, BS={BATCH_SIZE}")

    # --- DATA ---
    full_ds = MSDDataset(DATA_ROOT, TASK, IMG_SIZE, MODALITY, SUBSET_SIZE)
    train_len = int(TRAIN_SPLIT * len(full_ds))
    val_len = int(VAL_SPLIT * len(full_ds))
    test_len = len(full_ds) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Splits: train={train_len}, val={val_len}, test={test_len}")

    # --- PHYSICS ---
    physics = get_physics_operator(IMG_SIZE, ACCEL, CENTER_FRAC, DEVICE, modality=MODALITY)

    # ================================================================
    # PHASE 1: Train U-Net on Clean Ground Truth
    # ================================================================
    print("\n=== PHASE 1: Training on Clean Ground Truth ===")
    model = UNet().to(DEVICE)
    loss_fn = DiceBCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR_UNET)

    best_val_dice = 0.0
    ckpt_path = os.path.join(OUTPUT_DIR, "model_clean.pth")

    for ep in range(EPOCH_CLEAN):
        model.train()
        epoch_loss = 0.0
        for img, mask in train_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            # Normalize raw HU -> [0,1] only at U-Net input
            x_in = robust_normalize(img)
            opt.zero_grad()
            pred = model(x_in)
            loss = loss_fn(pred, mask)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # Validate on clean images
        model.eval()
        val_dice = 0.0
        for img, mask in val_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                pred = model(robust_normalize(img))
                val_dice += compute_dice(pred, mask).item()
        val_dice /= len(val_loader)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)

        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {ep+1:2d}/{EPOCH_CLEAN} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} | Best: {best_val_dice:.4f}")

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path))

    # Test Upper Bound
    model.eval()
    upper_dice = 0.0
    for img, mask in test_loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        with torch.no_grad():
            pred = model(robust_normalize(img))
            upper_dice += compute_dice(pred, mask).item()
    upper_dice /= len(test_loader)
    print(f"\n  Upper Bound (Clean Test): {upper_dice:.4f}")

    # ================================================================
    # PHASE 2: Lower Bound — Test Clean Model on Noisy FBP
    # ================================================================
    print("\n=== PHASE 2: Lower Bound (Noisy FBP) ===")
    model.eval()

    lower_dice = 0.0
    per_sample = []

    for i, (img, mask) in enumerate(test_loader):
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        # Forward through physics + noise (in raw HU space)
        y_clean = physics(img)
        y = y_clean + NOISE_SIGMA * torch.randn_like(y_clean)

        # FBP reconstruction (output is in raw HU space)
        with torch.no_grad():
            x_recon = physics.A_dagger(y)
            # Same normalization as clean: raw HU -> [0,1]
            x_in = robust_normalize(x_recon)
            pred = model(x_in)
            d = compute_dice(pred, mask).item()
            lower_dice += d
            per_sample.append(d)

    lower_dice /= len(test_loader)

    # ================================================================
    # RESULTS
    # ================================================================
    print(f"\n{'='*50}")
    print(f"  Upper Bound (Clean):  {upper_dice:.4f}")
    print(f"  Lower Bound (Noisy):  {lower_dice:.4f}")
    print(f"{'='*50}")

    if lower_dice >= 0.20:
        print(f"  PASS -- Lower Bound {lower_dice:.4f} >= 0.20 target")
    else:
        print(f"  FAIL -- Lower Bound {lower_dice:.4f} < 0.20 target")
        print(f"\n  Per-sample Dice scores:")
        for i, d in enumerate(per_sample):
            print(f"    Sample {i+1}: {d:.4f}")
        print(f"\n  Stats: min={min(per_sample):.4f}, max={max(per_sample):.4f}, "
              f"median={sorted(per_sample)[len(per_sample)//2]:.4f}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "lower_bound_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Upper Bound: {upper_dice:.4f}\n")
        f.write(f"Lower Bound: {lower_dice:.4f}\n")
        f.write(f"Target: 0.20\n")
        f.write(f"Pass: {lower_dice >= 0.20}\n")
        f.write(f"\nPer-sample:\n")
        for i, d in enumerate(per_sample):
            f.write(f"  {i+1}: {d:.4f}\n")
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
