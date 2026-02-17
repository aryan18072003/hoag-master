import torch
import torch.nn as nn
import numpy as np
import deepinv as dinv


# ==========================================
#  1. PHYSICS OPERATOR FACTORY
# ==========================================
def get_physics_operator(img_size, acceleration, center_frac, device, modality="CT"):

    if modality == "CT":
        # --- CT PHYSICS (Radon Transform) ---
        if acceleration == 1:
            num_views = 180 
        else:
            num_views = int(180 / acceleration) 
            
        angles = torch.linspace(0, 180, num_views).to(device)
        
        physics = dinv.physics.Tomography(
            angles=angles,
            img_width=img_size,
            circle=False,
            device=device
        )

        print(f"-> Computing Norm for Accel {acceleration}...")
        norm_val = physics.compute_norm(torch.randn(1, 1, img_size, img_size, device=device))
        print(f"   Norm found: {norm_val:.2f}. Applying normalization...")
        
        class NormalizedPhysics(dinv.physics.Physics):
            def __init__(self, original_physics, norm):
                super().__init__()
                self.original = original_physics
                self.norm_const = norm
                
            def A(self, x):
                return self.original.A(x) / self.norm_const
            
            def A_adjoint(self, y):
                return self.original.A_adjoint(y) / self.norm_const
            
            def A_dagger(self, y):
                return self.original.A_dagger(y * self.norm_const)
                
            def forward(self, x): 
                return self.A(x)

        physics = NormalizedPhysics(physics, norm_val)
        return physics

    elif modality == "MRI":
        mask = torch.zeros((1, img_size, img_size))
        
        pad = (img_size - int(img_size * center_frac) + 1) // 2
        width = max(1, int(img_size * center_frac))
        mask[:, :, pad:pad + width] = 1.0
        
        num_keep = int(img_size / acceleration)
        all_cols = np.arange(img_size)
        kept_cols = np.where(mask[0, 0, :].cpu().numpy() == 1)[0]
        zero_cols = np.setdiff1d(all_cols, kept_cols)
        
        if len(zero_cols) > 0 and (num_keep - len(kept_cols) > 0):
            chosen = np.random.choice(zero_cols, num_keep - len(kept_cols), replace=False)
            mask[:, :, chosen] = 1.0
            
        mask = mask.to(device)
        physics = dinv.physics.MRI(mask=mask, img_size=(1, img_size, img_size), device=device)
        return physics

    else:
        raise ValueError(f"Unsupported modality: {modality}")


# ==========================================
#  2. INNER LOSS FUNCTION (h in HOAG)
# ==========================================
def inner_loss_func(w, theta, y, physics_op):
    # --- DATA FIDELITY TERM: -----------
    fid = torch.norm(y - physics_op(w), p=2)**2
    
    # --- REGULARIZATION PARAMETERS ---
    reg_weight = torch.exp(theta[0].clamp(max=1.0))   # λ = exp(θ₀)
    eps = torch.exp(theta[1].clamp(min=-12.0))         # ε = exp(θ₁)
    
    # --- TOTAL VARIATION (TV) REGULARIZATION ---
    dx = torch.roll(w, 1, 2) - w   # Horizontal gradient
    dy = torch.roll(w, 1, 3) - w   # Vertical gradient
    
    tv_penalty = torch.mean(torch.sqrt(dx**2 + dy**2 + eps))
    
    return fid + reg_weight * tv_penalty


# ==========================================
#  3. NORMALIZATION UTILITY
# ==========================================
# def robust_normalize(x):
#     b = x.shape[0]
#     x_flat = x.view(b, -1)
    
#     val_min = torch.quantile(x_flat, 0.01, dim=1).view(b, 1, 1, 1)
#     val_max = torch.quantile(x_flat, 0.99, dim=1).view(b, 1, 1, 1)
    
#     x = torch.clamp(x, val_min, val_max)
    
#     denom = val_max - val_min
#     denom = torch.where(denom > 1e-7, denom, torch.ones_like(denom))
    
#     return (x - val_min) / denom


def robust_normalize(x):
    """
    Z-score normalization: (x - mean) / std.
    This is standard for neural networks and robust to scaling shifts.
    """
    # Calculate mean and std per image in the batch
    mean = x.flatten(1).mean(1).view(-1, 1, 1, 1)
    std = x.flatten(1).std(1).view(-1, 1, 1, 1)
    
    # Avoid division by zero
    std = torch.where(std > 1e-7, std, torch.ones_like(std))
    
    return (x - mean) / std