"""
main.py — Task-Driven Physics Optimization via HOAG
====================================================

Experiment: Automatically learn optimal regularization hyperparameters (θ)
for a physics-based CT reconstruction algorithm (Inverse Radon Transform)
by maximizing the performance of a downstream segmentation network (U-Net).

This uses the HOAG (Hyperparameter Optimization with Approximate Gradient)
bilevel optimization approach:
    Inner Problem: Solve Tikhonov/TV-regularized reconstruction
    Outer Problem: Minimize segmentation loss (Dice + BCE) via U-Net

The experiment runs 4 phases:
    Phase 1: Upper Bound — Train U-Net on clean ground truth (best possible)
    Phase 2: Lower Bound — Test clean U-Net on noisy reconstructions (worst case)
    Phase 3: Approach 1 — Fix U-Net, optimize θ only via HOAG
    Phase 4: Approach 2 — Joint optimization of θ + U-Net weights

Reference:
    Pedregosa, F. "Hyperparameter optimization with approximate gradient."
    ICML 2016. http://jmlr.org/proceedings/papers/v48/pedregosa16.html
"""

import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split

from models import UNet
from dataset import MSDDataset
from physics import get_physics_operator, inner_loss_func, robust_normalize

# Import the HOAG optimizer (new file — mirrors the original hoag/hoag.py)
from hoag import HOAGState, hoag_step, solve_inner_problem


# ==========================================
#        1. CONFIGURATION
# ==========================================
class Config:
    """
    Central configuration for the experiment.
    Physics settings simulate a fast, low-dose CT scan (sparse-view + noise).
    """
    DATA_ROOT = "./"
    TASK = "Task09_Spleen"
    OUTPUT_DIR = "./results_hoag_single_op"
    MODALITY = "CT"
    
    # Dataset Splits
    SUBSET_SIZE = 100
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    IMG_SIZE = 128
    BATCH_SIZE = 4
    
    # --- SINGLE PHYSICS SETTING (SPARSE) ---
    ACCEL = 16         # 16x Acceleration → only 180/16 = 11 projection views
    NOISE_SIGMA = 0.1  # 10% Gaussian noise on sinogram
    CENTER_FRAC = 0.08
    
    # --- INNER OPTIMIZATION SETTINGS ---
    # These control how precisely the reconstruction is solved.
    # HOAG's tolerance schedule will override the fixed step count
    # with adaptive early stopping (see hoag.py solve_inner_problem).
    INNER_STEPS = 100    # Max inner optimization steps
    INNER_LR = 0.02      # Adam learning rate for inner solver
    
    # --- OUTER OPTIMIZATION SETTINGS ---
    EPOCHS = 15
    LR_UNET = 1e-3       # Adam lr for U-Net weights
    LR_THETA = 1e-3      # Adam lr for hyperparameters θ
    
    # --- HOAG-SPECIFIC SETTINGS ---
    # (See original hoag/hoag.py lines 13-14 for reference)
    HOAG_EPSILON_TOL_INIT = 1e-3    # Initial tolerance (hoag.py default)
    HOAG_TOLERANCE_DECREASE = 'exponential'  # Schedule: 'exponential', 'quadratic', 'cubic'
    HOAG_DECREASE_FACTOR = 0.9      # Exponential decay factor (hoag.py default)
    HOAG_CG_MAX_ITER = 20           # Max CG iterations for linear system
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
#        2. HELPER FUNCTIONS
# ==========================================
class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss for segmentation.
    Dice Loss: Measures global overlap between prediction and ground truth.
    BCE Loss: Measures pixel-wise accuracy.
    
    This is the OUTER/VALIDATION loss g(w*(θ)) in HOAG's bilevel framework.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        # 1. BCE Loss (Pixel-wise accuracy)
        bce_loss = self.bce(inputs, targets)
        
        # 2. Soft Dice Loss (Global overlap)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        dice_loss = 1 - dice
        
        # Combine: 90% BCE + 10% Dice is standard weighting
        return 0.9 * bce_loss + 0.1 * dice_loss



def print_progress(epoch, batch, total_batches, loss, theta, info=""):
    """Print training progress with current hyperparameter values."""
    reg_val = torch.exp(theta[0]).item()
    eps_val = torch.exp(theta[1]).item()
    sys.stdout.write(f"\r[{info}] Ep {epoch+1} | Batch {batch+1}/{total_batches} | "
                     f"Loss: {loss:.4f} | Reg: {reg_val:.5f} | Smooth: {eps_val:.5f}")
    sys.stdout.flush()


def validate(model, val_loader, physics_op, theta=None, steps=0, mode="clean"):
    """
    Validation logic computing Dice score.
    
    Supports three modes matching the experiment phases:
        clean: Direct clean input → U-Net (Upper Bound)
        noisy: Noisy reconstruction → U-Net (Lower Bound)
        hoag:  HOAG-optimized reconstruction → U-Net (Approach 1 & 2)
    """
    model.eval()
    dice_score = 0.0
    
    for i, (img, mask) in enumerate(val_loader):
        img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
        
        # --- MODE 1: CLEAN (Upper Bound) ---
        if mode == "clean":
            x_in = robust_normalize(img)
        
        # --- MODE 2: NOISY (Lower Bound) ---
        elif mode == "noisy":
            y_clean = physics_op(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            with torch.no_grad():
                x_recon = physics_op.A_dagger(y)
                
            x_mag = torch.sqrt(x_recon[:,0:1]**2 + x_recon[:,1:2]**2 + 1e-8)
            x_in = robust_normalize(x_mag)

        # --- MODE 3: HOAG (Optimized Reconstruction) ---
        elif mode == "hoag":
            y_clean = physics_op(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # Solve inner problem with the HOAG-optimized theta
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

        # Predict segmentation mask
        with torch.no_grad():
            pred = (model(x_in) > 0.5).float()
            dice_score += (2. * (pred * mask).sum()) / (pred.sum() + mask.sum() + 1e-8)
            
    return dice_score.item() / len(val_loader)


# ==========================================
#        3. MAIN EXPERIMENT
# ==========================================
def run_experiment():
    print(f"--- Starting Experiment: {Config.TASK} (HOAG Bilevel Optimization) ---")
    print(f"    HOAG Settings: tol_init={Config.HOAG_EPSILON_TOL_INIT}, "
          f"schedule={Config.HOAG_TOLERANCE_DECREASE}, "
          f"decay={Config.HOAG_DECREASE_FACTOR}")
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
    # PHYSICS OPERATOR (Radon Transform for CT)
    # ====================================================================
    physics = get_physics_operator(Config.IMG_SIZE, Config.ACCEL, Config.CENTER_FRAC, Config.DEVICE, modality=Config.MODALITY)
    
    loss_fn = DiceBCELoss()
    results = {}
    dummy_theta = torch.tensor([-10.0, -10.0])  # ~zero regularization for baselines
    
    # ====================================================================
    # PHASE 1: UPPER BOUND — Train U-Net on Clean Ground Truth
    # ====================================================================
    # This establishes the theoretical maximum performance.
    # No physics simulation involved — U-Net sees perfect images.
    print("\n--- PHASE 1: Upper Bound (Training on Clean Ground Truth) ---")
    model_upper = UNet().to(Config.DEVICE)
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "model_upper_clean.pth")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    opt = torch.optim.Adam(model_upper.parameters(), lr=Config.LR_UNET)
    
    for ep in range(Config.EPOCHS):
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
    # Quantifies the domain gap: how much performance drops when the U-Net
    # (trained on clean images) receives noisy, sparse CT reconstructions.
    print("\n--- PHASE 2: Lower Bound (Testing Clean Model on Noisy Physics) ---")
    results['Lower Bound'] = validate(model_upper, test_loader, physics, theta=dummy_theta, mode="noisy")
    print(f" -> Final Lower Bound (Noisy): {results['Lower Bound']:.4f}")

    # ====================================================================
    # PHASE 3: APPROACH 1 — HOAG: Optimize θ Only (Fixed U-Net)
    # ====================================================================
    # This phase uses the HOAG bilevel optimization algorithm:
    #   - The U-Net weights are FROZEN (from Phase 1)
    #   - Only the physics hyperparameters θ = [log(λ), log(ε)] are optimized
    #   - θ controls the TV regularization in the inner reconstruction problem
    #
    # HOAG Algorithm per Batch:
    #   1. Solve inner problem: w* = argmin_w ||y-Aw||² + exp(θ₀)·TV(w, exp(θ₁))
    #      with tolerance-based stopping (HOAG's key innovation)
    #   2. Compute outer loss: g = DiceBCE(UNet(w*), mask)
    #   3. Compute hypergradient dg/dθ via implicit differentiation:
    #      dg/dθ = ∂g/∂θ - (∂g/∂w) · H⁻¹ · (∂²h/∂w∂θ)
    #   4. Update θ using Adam (with HOAG-computed gradient)
    #
    # The tolerance decreases each iteration (HOAG's efficiency trick):
    #   Early: solve roughly → fast but imprecise hypergradient
    #   Late:  solve precisely → slow but accurate hypergradient
    print("\n--- PHASE 3: Approach 1 (HOAG — Optimizing Theta Only) ---")
    
    # Load the clean model and FREEZE all weights
    # This is the "fixed critic" — we only optimize the physics parameters
    model_fixed = UNet().to(Config.DEVICE)
    model_fixed.load_state_dict(torch.load(ckpt_path)) 
    model_fixed.train()  # Keep in train mode for BatchNorm statistics
    for p in model_fixed.parameters():
        p.requires_grad = False  # Freeze all U-Net weights
    
    # Initialize θ in log-space:
    #   θ[0] = log(λ) ≈ -4.6 → λ ≈ 0.01 (regularization weight)
    #   θ[1] = log(ε) ≈ -5.0 → ε ≈ 0.007 (TV smoothing parameter)
    theta = torch.tensor([-4.6, -5.0], device=Config.DEVICE).requires_grad_(True)
    
    # Adam optimizer for θ — receives gradients computed by HOAG
    # (We use Adam instead of HOAG's simple 1/L step-size for better convergence)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
    # Initialize HOAG State — tracks tolerance schedule and CG warm-start
    # This is the PyTorch equivalent of the state variables in hoag.py lines 82-98:
    #   epsilon_tol, Bxk (warm-start), g_func_old, norm_init
    hoag_state = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )
    
    path_hoag = os.path.join(Config.OUTPUT_DIR, "hoag_theta.pth")

    for ep in range(Config.EPOCHS): 
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            # --- Simulate Noisy Sparse-View CT Scan ---
            # y = A·x + η  (forward projection + noise)
            y_clean = physics(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # --- HOAG STEP ---
            # This single call performs the complete HOAG outer iteration:
            #   1. Solves inner problem to current tolerance (w*)
            #   2. Computes Hessian-vector products via autograd
            #   3. Solves H·q = ∂g/∂w via CG (with warm-start)
            #   4. Computes hypergradient via implicit differentiation
            #   5. Decreases tolerance for next iteration
            # See hoag.py hoag_step() for detailed documentation.
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
            
            # --- Update θ using Adam with the HOAG-computed hypergradient ---
            # The hypergradient is clamped to [-1, 1] for stability
            # (prevents explosive updates from noisy CG solutions)
            opt_theta.zero_grad()
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            
            # Project θ to valid range (similar to original hoag.py lines 196-197)
            # θ[0] ∈ [-9, -2] → λ = exp(θ₀) ∈ [0.00012, 0.135]
            # θ[1] ∈ [-12, -2] → ε = exp(θ₁) ∈ [6e-6, 0.135]
            with torch.no_grad():
                theta[0].clamp_(-9.0, -2.0)
                theta[1].clamp_(-12.0, -2.0)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 1 (HOAG)")
        
        print(f"  [ε_tol: {hoag_state.epsilon_tol:.2e}]")
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state.epsilon_tol}, path_hoag)

    results['Approach 1'] = validate(model_fixed, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    print(f" -> Final Approach 1 Score: {results['Approach 1']:.4f}")

    # ====================================================================
    # PHASE 4: APPROACH 2 — HOAG Joint Learning (θ + U-Net)
    # ====================================================================
    # This phase performs JOINT optimization:
    #   - Both the physics parameters θ AND the U-Net weights are trained
    #   - θ is updated via HOAG hypergradient (same as Phase 3)
    #   - U-Net weights are updated via standard backprop on the task loss
    #
    # Alternating optimization per batch:
    #   Step A: Fix θ → train U-Net on w*(θ) reconstructions (standard SGD/Adam)
    #   Step B: Fix U-Net → compute HOAG hypergradient → update θ (Adam)
    #
    # This allows the U-Net to ADAPT to the reconstruction quality,
    # while the reconstruction adapts to what the U-Net needs.
    print("\n--- PHASE 4: Approach 2 (Joint Learning — Theta + U-Net) ---")
    
    # Start from the clean model (Phase 1 weights) — fine-tune jointly
    model_joint = UNet().to(Config.DEVICE)
    model_joint.load_state_dict(torch.load(ckpt_path))
    opt_model = torch.optim.Adam(model_joint.parameters(), lr=Config.LR_UNET)
    
    # Start from the best θ found in Phase 3 (warm-start)
    theta = torch.load(path_hoag)['theta'].to(Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
    # Fresh HOAG state for Phase 4
    # (Reset tolerance — the joint problem is different from Phase 3)
    hoag_state_joint = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )
    
    path_joint = os.path.join(Config.OUTPUT_DIR, "model_joint.pth")
    path_theta_joint = os.path.join(Config.OUTPUT_DIR, "theta_joint.pth")

    for ep in range(Config.EPOCHS):
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            # Simulate noisy sparse-view CT scan
            y_clean = physics(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # ============================================================
            # STEP A: Update U-Net Weights (Standard Backprop)
            # ============================================================
            # Solve inner problem for current θ to get reconstruction w*
            # Then train U-Net on this reconstruction (no HOAG needed here)
            
            # Inner solve using HOAG's tolerance-based stopping
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
            
            # Detach w* from inner graph — U-Net training doesn't flow gradients to θ
            w_fixed = w_for_unet.detach().clone().requires_grad_(False)
            x_in = robust_normalize(torch.sqrt(w_fixed[:,0:1]**2 + w_fixed[:,1:2]**2 + 1e-8))
            
            model_joint.train()
            opt_model.zero_grad()
            loss_unet = loss_fn(model_joint(x_in), mask)
            loss_unet.backward()
            opt_model.step()
            
            # ============================================================
            # STEP B: Update θ via HOAG Hypergradient
            # ============================================================
            # Freeze model for hypergradient calculation
            # (We want dg/dθ with fixed U-Net weights for this step)
            model_joint.eval()
            
            # HOAG step computes the full hypergradient:
            #   dg/dθ = ∂g/∂θ - (∂g/∂w)·H⁻¹·(∂²h/∂w∂θ)
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
            
            # Update θ using Adam with HOAG-computed hypergradient
            opt_theta.zero_grad()
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            
            # Project θ to valid range (original hoag.py lines 196-197)
            with torch.no_grad():
                theta[0].clamp_(-9.0, -2.0)
                theta[1].clamp_(-12.0, -2.0)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 2 (Joint)")
        
        print(f"  [ε_tol: {hoag_state_joint.epsilon_tol:.2e}]")
        torch.save(model_joint.state_dict(), path_joint)
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state_joint.epsilon_tol}, path_theta_joint)

    results['Approach 2'] = validate(model_joint, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    
    # ====================================================================
    # FINAL RESULTS
    # ====================================================================
    print("\n=== FINAL RESULTS ===")
    print(f"1. Upper Bound: {results['Upper Bound']:.4f}")
    print(f"2. Lower Bound: {results['Lower Bound']:.4f}")
    print(f"3. Approach 1:  {results['Approach 1']:.4f}")
    print(f"4. Approach 2:  {results['Approach 2']:.4f}")

    # Save results to text file
    results_path = os.path.join(Config.OUTPUT_DIR, "final_results.txt")
    with open(results_path, "w") as f:
        f.write("=== FINAL RESULTS ===\n")
        f.write(f"Task: {Config.TASK}\n")
        f.write(f"Accel: {Config.ACCEL}x | Noise: {Config.NOISE_SIGMA}\n")
        f.write(f"HOAG: tol_init={Config.HOAG_EPSILON_TOL_INIT}, "
                f"schedule={Config.HOAG_TOLERANCE_DECREASE}, "
                f"decay={Config.HOAG_DECREASE_FACTOR}\n")
        f.write(f"Epochs: {Config.EPOCHS} | Inner Steps: {Config.INNER_STEPS}\n\n")
        f.write(f"1. Upper Bound (Clean):      {results['Upper Bound']:.4f}\n")
        f.write(f"2. Lower Bound (Noisy):      {results['Lower Bound']:.4f}\n")
        f.write(f"3. Approach 1 (HOAG theta):  {results['Approach 1']:.4f}\n")
        f.write(f"4. Approach 2 (HOAG joint):  {results['Approach 2']:.4f}\n")
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    run_experiment()