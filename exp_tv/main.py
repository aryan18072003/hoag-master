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
from physics import get_physics_operator, inner_loss_func, robust_normalize

from hoag import HOAGState, hoag_step, solve_inner_problem


# ==========================================
#        1. CONFIGURATION
# ==========================================
class Config:
    """
    Central configuration for the experiment.
    Physics settings simulate a fast, low-dose CT scan (sparse-view + noise).
    """
    DATA_ROOT = "../ct_data"
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
    ACCEL = 6         
    NOISE_SIGMA = 0.5  
    CENTER_FRAC = 0.08
    
    # --- INNER OPTIMIZATION SETTINGS ---
    INNER_STEPS = 500   
    INNER_LR = 0.1     
    
    # --- OUTER OPTIMIZATION SETTINGS ---
    EPOCH_CLEAN = 50
    EPOCHS = 25               
    LR_UNET = 5e-3
    LR_THETA = 0.05           
    
    # --- HOAG-SPECIFIC SETTINGS ---
    HOAG_EPSILON_TOL_INIT = 1e-3
    HOAG_TOLERANCE_DECREASE = 'exponential'
    HOAG_DECREASE_FACTOR = 0.9
    HOAG_CG_MAX_ITER = 50  
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm(img):
    img = torch.clamp(img, min=-150, max=250)
    img = (img + 150) / 400.0
    return img
# ==========================================
#        2. HELPER FUNCTIONS
# ==========================================
class DiceBCELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        # 1. BCE Loss (Pixel-wise accuracy)
        bce_loss = self.bce(inputs, targets)
        
        # 2. Soft Dice Loss
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        dice_loss = 1 - dice
        
        # Combine:
        return 0.9 * bce_loss + 0.1 * dice_loss



def print_progress(epoch, batch, total_batches, loss, theta, info=""):
    reg_val = torch.exp(theta[0]).item()
    eps_val = torch.exp(theta[1]).item()
    sys.stdout.write(f"\r[{info}] Ep {epoch+1} | Batch {batch+1}/{total_batches} | "
                     f"Loss: {loss:.4f} | Reg: {reg_val:.5f} | Smooth: {eps_val:.5f}")
    sys.stdout.flush()


def validate(model, val_loader, physics_op, theta=None, steps=0, mode="clean"):
    model.eval()
    dice_score = 0.0
    
    for i, (img, mask) in enumerate(val_loader):
        img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
        
        # --- MODE 1: CLEAN (Upper Bound) ---
        if mode == "clean":
            x_in = norm(img)
        
        # --- MODE 2: NOISY (Lower Bound) ---
        elif mode == "noisy":
            y_clean = physics_op(img)
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            with torch.no_grad():
                x_recon = physics_op.A_dagger(y)
                
            x_in = norm(x_recon)

        # --- MODE 3: HOAG (Optimized Reconstruction) ---
        elif mode == "hoag":
            y_clean = physics_op(img)
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # Solve inner problem with the HOAG-optimized theta
            w = physics_op.A_dagger(y).detach().clone()
            w.requires_grad_(True)
            optimizer_inner = torch.optim.Adam([w], lr=Config.INNER_LR)
            scheduler_inner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_inner, T_max=steps, eta_min=Config.INNER_LR * 0.01)
            
            with torch.enable_grad():
                for _ in range(steps):
                    optimizer_inner.zero_grad()
                    loss = inner_loss_func(w, theta, y, physics_op)
                    loss.backward()
                    optimizer_inner.step()
                    scheduler_inner.step()     
            x_recon = w.detach()
            
            x_in = norm(x_recon)

        # Predict segmentation mask
        with torch.no_grad():
            pred = (model(x_in) > 0.5).float()
            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum()
            dice_score += (2. * intersection + 1e-6) / (union + 1e-6)
            
    return dice_score.item() / len(val_loader)


# ==========================================
#        3. MAIN EXPERIMENT
# ==========================================
def run_experiment():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    
    loss_fn = torch.nn.BCELoss()
    results = {}
    dummy_theta = torch.tensor([-10.0, -10.0])  # ~zero regularization for baselines
    
    # ====================================================================
    # PHASE 1: UPPER BOUND — Train U-Net on Clean Ground Truth
    # ====================================================================
    print("\n--- PHASE 1: Upper Bound (Training on Clean Ground Truth) ---")
    model_upper = UNet().to(Config.DEVICE)
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "model_upper_clean.pth")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    opt = torch.optim.Adam(model_upper.parameters(), lr=Config.LR_UNET)
    
    for ep in range(Config.EPOCH_CLEAN):
        model_upper.train()
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            x_in = norm(img)
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
    print(f" -> Final Lower Bound (Noisy): {results['Lower Bound']:.4f}")

    # ====================================================================
    # PHASE 3: APPROACH 1 — HOAG: Optimize theta Only (Fixed U-Net)
    # ====================================================================
    print("\n--- PHASE 3: Approach 1 (HOAG — Optimizing Theta Only) ---")
    
    model_fixed = UNet().to(Config.DEVICE)
    model_fixed.load_state_dict(torch.load(ckpt_path)) 
    model_fixed.eval()
    for p in model_fixed.parameters():
        p.requires_grad = False
    
    theta = torch.tensor([-1.0, -4.0], device=Config.DEVICE).requires_grad_(True)
    
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
    # Initialize HOAG State
    hoag_state = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )
    
    path_hoag = os.path.join(Config.OUTPUT_DIR, "hoag_theta.pth")

    for ep in range(Config.EPOCHS): 
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            # Simulate Noisy Sparse-View CT Scan
            y_clean = physics(img)
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # HOAG STEP
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
            
            # Update theta using Adam with the HOAG-computed hypergradient
            opt_theta.zero_grad()
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            
            with torch.no_grad():
                theta[0].clamp_(-9.0, 1.0)
                theta[1].clamp_(-12.0, -2.0)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 1 (HOAG)")
        
        print(f"  [epsilon_tol: {hoag_state.epsilon_tol:.2e}]")
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state.epsilon_tol}, path_hoag)

    results['Approach 1'] = validate(model_fixed, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    print(f" -> Final Approach 1 Score: {results['Approach 1']:.4f}")

    # ====================================================================
    # PHASE 4: APPROACH 2 — HOAG Joint Learning (theta + U-Net)
    # ====================================================================
    print("\n--- PHASE 4: Approach 2 (Joint Learning — Theta + U-Net) ---")
    
    model_joint = UNet().to(Config.DEVICE)
    #model_joint.load_state_dict(torch.load(ckpt_path))
    opt_model = torch.optim.Adam(model_joint.parameters(), lr=Config.LR_UNET)
    
    #theta = torch.load(path_hoag)['theta'].to(Config.DEVICE).requires_grad_(True)
    theta = torch.tensor([-1.0, -4.0], device=Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
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
            y_clean = physics(img)
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # ============================================================
            # STEP A: Solve Inner Problem ONCE (shared between U-Net and theta updates)
            # ============================================================
            w_star, _ = solve_inner_problem(
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
            
            # ============================================================
            # STEP B: Update U-Net Weights (Standard Backprop)
            # ============================================================
            w_fixed = w_star.detach().clone().requires_grad_(False)
            x_in = norm(w_fixed)
            
            model_joint.train()
            opt_model.zero_grad()
            loss_unet = loss_fn(model_joint(x_in), mask)
            loss_unet.backward()
            opt_model.step()
            
            # STEP C: Update theta via HOAG Hypergradient
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
            
            # Update theta using Adam with HOAG-computed hypergradient
            opt_theta.zero_grad()
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            
            # Project theta to valid range
            with torch.no_grad():
                theta[0].clamp_(-9.0, 1.0)
                theta[1].clamp_(-12.0, -2.0)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 2 (Joint)")
        
        print(f"  [epsilon_tol: {hoag_state_joint.epsilon_tol:.2e}]")
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