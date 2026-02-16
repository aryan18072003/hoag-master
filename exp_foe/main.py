"""
main.py — Task-Driven Physics Optimization via HOAG (FoE Regularizer)
=====================================================================

Same experiment as my_exp/main.py but using Field of Experts (FoE)
as the regularizer instead of smooth Total Variation (TV).

Key differences from my_exp/main.py:
    1. θ has 50 parameters (5 weights + 45 filter coefficients) vs 2
    2. θ is initialized with derivative-like filters
    3. Clamping applies only to weights, not filter coefficients
    4. Progress display shows mean weight and filter norm

The 4-phase structure is identical:
    Phase 1: Upper Bound — Train U-Net on clean ground truth
    Phase 2: Lower Bound — Test clean U-Net on noisy reconstructions
    Phase 3: Approach 1 — Fix U-Net, optimize θ only via HOAG
    Phase 4: Approach 2 — Joint optimization of θ + U-Net weights
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split

from models import UNet
from dataset import MSDDataset
from physics import (get_physics_operator, inner_loss_func, robust_normalize,
                     initialize_theta, parse_theta, NUM_EXPERTS,
                     FILTER_SIZE, IN_CHANNELS, THETA_SIZE)

# Import the HOAG optimizer (same as my_exp — algorithm is regularizer-agnostic)
from hoag import HOAGState, hoag_step, solve_inner_problem


# ==========================================
#        1. CONFIGURATION
# ==========================================
class Config:
    """
    Central configuration for the FoE experiment.
    Physics settings are identical to my_exp.
    FoE-specific settings are at the bottom.
    """
    DATA_ROOT = "./"
    TASK = "Task09_Spleen"
    OUTPUT_DIR = "./results_hoag_foe"
    MODALITY = "CT"
    
    # Dataset Splits (same as my_exp)
    SUBSET_SIZE = 100
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    IMG_SIZE = 128
    BATCH_SIZE = 4
    
    # --- SINGLE PHYSICS SETTING (SPARSE) ---
    ACCEL = 16
    NOISE_SIGMA = 0.1
    CENTER_FRAC = 0.08
    
    # --- INNER OPTIMIZATION SETTINGS ---
    INNER_STEPS = 100
    INNER_LR = 0.02
    
    # --- OUTER OPTIMIZATION SETTINGS ---
    EPOCHS_CLEAN = 50     # Phase 1: enough to fully converge on clean data
    EPOCHS_HOAG = 15      # Phase 3: HOAG theta optimization
    EPOCHS_JOINT = 10      # Phase 4: fine-tuning (short — model is already pretrained)
    LR_UNET = 1e-3
    LR_THETA = 1e-3   # Conservative for 261-param FoE
    
    # --- HOAG-SPECIFIC SETTINGS ---
    HOAG_EPSILON_TOL_INIT = 1e-3
    HOAG_TOLERANCE_DECREASE = 'exponential'
    HOAG_DECREASE_FACTOR = 0.995   # Per-batch: 0.995^18 ≈ 0.91/epoch (not 0.9^18 = 0.15!)
    HOAG_CG_MAX_ITER = 20
    
    # --- FoE-SPECIFIC SETTINGS (matching paper) ---
    NUM_EXPERTS = NUM_EXPERTS        # J=5 expert filters
    FILTER_SIZE = FILTER_SIZE        # 5×5 kernels
    IN_CHANNELS = IN_CHANNELS        # 2 channels (CT complex)
    THETA_SIZE = THETA_SIZE          # 261 total parameters
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
#        2. HELPER FUNCTIONS
# ==========================================
class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation (same as my_exp)."""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        dice_loss = 1 - dice
        return 0.9 * bce_loss + 0.1 * dice_loss


def print_progress(epoch, batch, total_batches, loss, theta, info=""):
    """Print training progress with paper-FoE hyperparameter summary."""
    global_w, filt_w, smooth_p, filters = parse_theta(theta)
    gw = torch.exp(global_w).item()
    mean_fw = torch.exp(filt_w).mean().item()
    mean_nu = torch.exp(smooth_p).mean().item()
    fn = filters.norm().item()
    sys.stdout.write(f"\r[{info}] Ep {epoch+1} | Batch {batch+1}/{total_batches} | "
                     f"Loss: {loss:.4f} | GW: {gw:.3f} | FW: {mean_fw:.4f} | ν: {mean_nu:.4f} | FN: {fn:.2f}")
    sys.stdout.flush()


def validate(model, val_loader, physics_op, theta=None, steps=0, mode="clean"):
    """
    Validation logic computing Dice score.
    Identical to my_exp except using FoE inner loss.
    """
    model.eval()
    dice_score = 0.0
    
    for i, (img, mask) in enumerate(val_loader):
        img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
        
        if mode == "clean":
            x_in = robust_normalize(img)
        
        elif mode == "noisy":
            y_clean = physics_op(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            with torch.no_grad():
                x_recon = physics_op.A_dagger(y)
            x_mag = torch.sqrt(x_recon[:,0:1]**2 + x_recon[:,1:2]**2 + 1e-8)
            x_in = robust_normalize(x_mag)

        elif mode == "hoag":
            y_clean = physics_op(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            w = physics_op.A_dagger(y).detach().clone()
            w.requires_grad_(True)
            optimizer_inner = torch.optim.Adam([w], lr=Config.INNER_LR)
            
            with torch.enable_grad():
                for _ in range(steps):
                    optimizer_inner.zero_grad()
                    loss = inner_loss_func(w, theta, y, physics_op)
                    loss.backward()
                    optimizer_inner.step()
                    with torch.no_grad(): w.clamp_(0.0, 1.0)
            x_recon = w.detach()
            
            x_mag = torch.sqrt(x_recon[:,0:1]**2 + x_recon[:,1:2]**2 + 1e-8)
            x_in = robust_normalize(x_mag)

        with torch.no_grad():
            pred = (model(x_in) > 0.5).float()
            dice_score += (2. * (pred * mask).sum()) / (pred.sum() + mask.sum() + 1e-8)
            
    return dice_score.item() / len(val_loader)


# ==========================================
#        3. θ CLAMPING FOR FoE
# ==========================================
def clamp_theta(theta):
    """
    Project θ to valid range for paper's FoE.
    
    Clamping:
        - θ₀ (global weight): [-6, 4]     (exp range: [0.002, 54.6])
        - θⱼ (filter weights): [-6, 4]     (exp range: [0.002, 54.6])
        - νⱼ (smoothing): [-8, 2]           (exp range: [0.0003, 7.4])
        - Filter coefficients: NOT clamped  (learned freely)
    """
    with torch.no_grad():
        # Global weight
        theta[0].clamp_(-6.0, 4.0)
        # Per-filter weights
        theta[1:1 + Config.NUM_EXPERTS].clamp_(-6.0, 4.0)
        # Smoothing params
        theta[1 + Config.NUM_EXPERTS : 1 + 2 * Config.NUM_EXPERTS].clamp_(-8.0, 2.0)


# ==========================================
#        4. MAIN EXPERIMENT
# ==========================================
def run_experiment():
    # Set seed for reproducible results (same data splits across runs)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"--- Starting Experiment: {Config.TASK} (HOAG + FoE Regularizer) ---")
    print(f"    FoE: {Config.NUM_EXPERTS} experts, {Config.FILTER_SIZE}x{Config.FILTER_SIZE} filters, "
          f"{Config.THETA_SIZE} total params")
    print(f"    HOAG: tol_init={Config.HOAG_EPSILON_TOL_INIT}, "
          f"schedule={Config.HOAG_TOLERANCE_DECREASE}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # ====================================================================
    # DATA SETUP
    # ====================================================================
    full_ds = MSDDataset(Config.DATA_ROOT, Config.TASK, Config.IMG_SIZE, Config.MODALITY, Config.SUBSET_SIZE)
    train_len = int(Config.TRAIN_SPLIT * len(full_ds))
    val_len   = int(Config.VAL_SPLIT * len(full_ds))
    test_len  = len(full_ds) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)
    
    # ====================================================================
    # PHYSICS OPERATOR
    # ====================================================================
    physics = get_physics_operator(Config.IMG_SIZE, Config.ACCEL, Config.CENTER_FRAC, Config.DEVICE, modality=Config.MODALITY)
    
    loss_fn = DiceBCELoss()
    results = {}
    # Dummy theta for phases that don't use HOAG
    dummy_theta = torch.zeros(Config.THETA_SIZE, device=Config.DEVICE)
    
    # ====================================================================
    # PHASE 1: UPPER BOUND — Train U-Net on Clean Ground Truth
    # ====================================================================
    print("\n--- PHASE 1: Upper Bound (Training on Clean Ground Truth) ---")
    model_upper = UNet().to(Config.DEVICE)
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "model_upper_clean.pth")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    opt = torch.optim.Adam(model_upper.parameters(), lr=Config.LR_UNET)
    
    for ep in range(Config.EPOCHS_CLEAN):
        model_upper.train()
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            x_in = robust_normalize(img)
            opt.zero_grad()
            pred = model_upper(x_in)
            loss = loss_fn(pred, mask)
            loss.backward()
            opt.step()
            print_progress(ep, i, len(train_loader), loss.item(), dummy_theta, "Clean Training")
    print(""); torch.save(model_upper.state_dict(), ckpt_path)

    results['Upper Bound'] = validate(model_upper, test_loader, physics, theta=dummy_theta, mode="clean")
    print(f" -> Final Upper Bound (Clean): {results['Upper Bound']:.4f}")

    # ====================================================================
    # PHASE 2: LOWER BOUND — Test Clean Model on Noisy Physics
    # ====================================================================
    print("\n--- PHASE 2: Lower Bound (Testing Clean Model on Noisy Physics) ---")
    results['Lower Bound'] = validate(model_upper, test_loader, physics, theta=dummy_theta, mode="noisy")
    print(f" -> Final Lower Bound (Noisy): {results['Lower Bound']:.6f}")

    # ====================================================================
    # PHASE 3: APPROACH 1 — HOAG: Optimize θ Only (Fixed U-Net)
    # ====================================================================
    # Same structure as my_exp Phase 3, but θ now has 50 FoE parameters
    print("\n--- PHASE 3: Approach 1 (HOAG + FoE — Optimizing Theta Only) ---")
    
    model_fixed = UNet().to(Config.DEVICE)
    model_fixed.load_state_dict(torch.load(ckpt_path)) 
    model_fixed.train()
    for p in model_fixed.parameters():
        p.requires_grad = False
    
    # Initialize θ with derivative-like FoE filters
    theta = initialize_theta(Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
    hoag_state = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )
    
    path_hoag = os.path.join(Config.OUTPUT_DIR, "hoag_theta_foe.pth")

    for ep in range(Config.EPOCHS_HOAG): 
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            y_clean = physics(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            hyper_grad, val_loss_value, w_star = hoag_step(
                theta=theta,
                y=y,
                physics_op=physics,
                model=model_fixed,
                loss_fn=loss_fn,
                mask=mask,
                inner_loss_fn=inner_loss_func,
                state=hoag_state,
                inner_lr=Config.INNER_LR,
                inner_steps=Config.INNER_STEPS,
                cg_max_iter=Config.HOAG_CG_MAX_ITER,
                verbose=0
            )
            
            opt_theta.zero_grad()
            # Norm-based gradient clipping (preserves direction, unlike element-wise clamp)
            grad_norm = hyper_grad.norm()
            max_grad_norm = 10.0  # Higher for 261-dim θ
            if grad_norm > max_grad_norm:
                hyper_grad = hyper_grad * (max_grad_norm / grad_norm)
            theta.grad = hyper_grad
            opt_theta.step()
            clamp_theta(theta)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 1 (FoE)")
        
        print(f"  [eps_tol: {hoag_state.epsilon_tol:.2e}]")
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state.epsilon_tol}, path_hoag)

    results['Approach 1'] = validate(model_fixed, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    print(f" -> Final Approach 1 Score: {results['Approach 1']:.4f}")

    # ====================================================================
    # PHASE 4: APPROACH 2 — HOAG Joint Learning (θ + U-Net)
    # ====================================================================
    print("\n--- PHASE 4: Approach 2 (Joint Learning — FoE Theta + U-Net) ---")
    
    model_joint = UNet().to(Config.DEVICE)
    model_joint.load_state_dict(torch.load(ckpt_path))
    opt_model = torch.optim.Adam(model_joint.parameters(), lr=Config.LR_UNET)
    
    # Warm-start θ from Phase 3
    theta = torch.load(path_hoag)['theta'].to(Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
    hoag_state_joint = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )
    
    path_joint = os.path.join(Config.OUTPUT_DIR, "model_joint_foe.pth")
    path_theta_joint = os.path.join(Config.OUTPUT_DIR, "theta_joint_foe.pth")

    for ep in range(Config.EPOCHS_JOINT):
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            y_clean = physics(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # STEP A: Update U-Net Weights
            w_for_unet, _ = solve_inner_problem(
                w_init=physics.A_dagger(y).detach().clone(),
                theta=theta,
                y=y,
                physics_op=physics,
                inner_loss_fn=inner_loss_func,
                state=hoag_state_joint,
                lr=Config.INNER_LR,
                max_steps=Config.INNER_STEPS,
                verbose=0
            )
            
            w_fixed = w_for_unet.detach().clone().requires_grad_(False)
            x_in = robust_normalize(torch.sqrt(w_fixed[:,0:1]**2 + w_fixed[:,1:2]**2 + 1e-8))
            
            model_joint.train()
            opt_model.zero_grad()
            loss_unet = loss_fn(model_joint(x_in), mask)
            loss_unet.backward()
            opt_model.step()
            
            # STEP B: Update θ via HOAG Hypergradient
            model_joint.eval()
            
            hyper_grad, val_loss_value, w_star = hoag_step(
                theta=theta,
                y=y,
                physics_op=physics,
                model=model_joint,
                loss_fn=loss_fn,
                mask=mask,
                inner_loss_fn=inner_loss_func,
                state=hoag_state_joint,
                inner_lr=Config.INNER_LR,
                inner_steps=Config.INNER_STEPS,
                cg_max_iter=Config.HOAG_CG_MAX_ITER,
                verbose=0
            )
            
            opt_theta.zero_grad()
            # Norm-based gradient clipping (preserves direction, unlike element-wise clamp)
            grad_norm = hyper_grad.norm()
            max_grad_norm = 10.0  # Higher for 261-dim θ
            if grad_norm > max_grad_norm:
                hyper_grad = hyper_grad * (max_grad_norm / grad_norm)
            theta.grad = hyper_grad
            opt_theta.step()
            clamp_theta(theta)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 2 (Joint)")
        
        print(f"  [eps_tol: {hoag_state_joint.epsilon_tol:.2e}]")
        torch.save(model_joint.state_dict(), path_joint)
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state_joint.epsilon_tol}, path_theta_joint)

    results['Approach 2'] = validate(model_joint, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    
    # ====================================================================
    # FINAL RESULTS
    # ====================================================================
    print("\n=== FINAL RESULTS (FoE) ===")
    print(f"1. Upper Bound: {results['Upper Bound']:.4f}")
    print(f"2. Lower Bound: {results['Lower Bound']:.4f}")
    print(f"3. Approach 1:  {results['Approach 1']:.4f}")
    print(f"4. Approach 2:  {results['Approach 2']:.4f}")

    # Save results to text file
    results_path = os.path.join(Config.OUTPUT_DIR, "final_results.txt")
    with open(results_path, "w") as f:
        f.write("=== FINAL RESULTS (FoE Regularizer) ===\n")
        f.write(f"Task: {Config.TASK}\n")
        f.write(f"Accel: {Config.ACCEL}x | Noise: {Config.NOISE_SIGMA}\n")
        f.write(f"FoE: {Config.NUM_EXPERTS} experts, {Config.FILTER_SIZE}x{Config.FILTER_SIZE} filters, "
                f"{Config.THETA_SIZE} params\n")
        f.write(f"HOAG: tol_init={Config.HOAG_EPSILON_TOL_INIT}, "
                f"schedule={Config.HOAG_TOLERANCE_DECREASE}, "
                f"decay={Config.HOAG_DECREASE_FACTOR}\n")
        f.write(f"Epochs: Clean={Config.EPOCHS_CLEAN}, HOAG={Config.EPOCHS_HOAG}, Joint={Config.EPOCHS_JOINT} | Inner Steps: {Config.INNER_STEPS}\n\n")
        f.write(f"1. Upper Bound (Clean):      {results['Upper Bound']:.4f}\n")
        f.write(f"2. Lower Bound (Noisy):      {results['Lower Bound']:.4f}\n")
        f.write(f"3. Approach 1 (HOAG theta):  {results['Approach 1']:.4f}\n")
        f.write(f"4. Approach 2 (HOAG joint):  {results['Approach 2']:.4f}\n")
        
        # Also save final θ values for analysis
        f.write(f"\n--- Final Theta ---\n")
        global_w, filt_w, smooth_p, filters = parse_theta(theta)
        f.write(f"Global weight: e^{global_w.item():.4f} = {torch.exp(global_w).item():.6f}\n")
        for k in range(Config.NUM_EXPERTS):
            f.write(f"Expert {k+1}: weight=e^{filt_w[k].item():.4f}={torch.exp(filt_w[k]).item():.6f}, "
                    f"nu={torch.exp(smooth_p[k]).item():.6f}\n")
            f.write(f"  Filter:\n{filters[k].detach().cpu().numpy()}\n")
    
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    run_experiment()
