"""
physics.py — CT/MRI Physics Simulation Layer
=============================================

This module simulates the physics of medical imaging:
    - Forward model A (Radon Transform for CT, Fourier sampling for MRI)
    - Adjoint model A^T (Backprojection for CT)
    - Inner loss function h(w, θ, y, A) for the HOAG bilevel problem

In the HOAG framework, this module defines the INNER OBJECTIVE:
    h(w, θ) = ||y - Aw||² + exp(θ₀) · TV(w, ε=exp(θ₁))
    
    where:
        w = reconstructed image (inner optimization variable)
        θ = [log(λ), log(ε)] are the learned hyperparameters
        A = physics forward operator (Radon transform)
        y = noisy sinogram measurement
        TV = isotropic Total Variation regularization

This corresponds to the training loss h(x, λ) in the original HOAG (hoag.py),
with the key difference being:
    - Original HOAG: h = logistic loss + L2 regularization
    - This project:   h = data fidelity + TV regularization

Dependencies:
    - deepinv: Provides differentiable physics operators (Tomography, MRI)
"""

import torch
import torch.nn as nn
import numpy as np
import deepinv as dinv


# ==========================================
#  1. PHYSICS OPERATOR FACTORY
# ==========================================
def get_physics_operator(img_size, acceleration, center_frac, device, modality="CT"):
    """
    Creates a differentiable physics forward model for CT or MRI.
    
    For CT (Radon Transform):
        - Number of projection views = 180 / acceleration
        - 16x acceleration → only 11 views (vs 180 full)
        - The operator is wrapped in NormalizedPhysics to ensure ||A|| ≈ 1.0
          This prevents numerical instability during optimization.
    
    Args:
        img_size: Image dimensions (square)
        acceleration: Undersampling factor (1 = full, 16 = sparse)
        center_frac: Center fraction for MRI k-space sampling
        device: PyTorch device (cuda/cpu)
        modality: "CT" or "MRI"
    
    Returns:
        physics: Normalized physics operator with .A() and .A_adjoint() methods
    """
    if modality == "CT":
        # --- CT PHYSICS (Radon Transform) ---
        if acceleration == 1:
            num_views = 180 
        else:
            num_views = int(180 / acceleration)  # e.g., 180/16 = 11 views
            
        angles = torch.linspace(0, 180, num_views).to(device)
        
        # Create the physics operator using DeepInverse library
        physics = dinv.physics.Tomography(
            angles=angles,
            img_width=img_size,
            circle=False,
            device=device
        )

        # === OPERATOR NORM NORMALIZATION ===
        # Compute ||A|| and divide both A and A^T by it.
        # This ensures ||A|| ≈ 1.0, which is critical because:
        #   - The inner loss has terms ||y - Aw||² and λ·TV(w)
        #   - If ||A|| >> 1, the data fidelity term dominates and λ becomes meaningless
        #   - If ||A|| ≈ 1, then λ directly controls the reconstruction quality
        #
        # This is analogous to the original HOAG README advice:
        #   "Standardize features... makes the Hessian better conditioned"
        print(f"-> Computing Norm for Accel {acceleration}...")
        norm_val = physics.compute_norm(torch.randn(1, 1, img_size, img_size, device=device))
        print(f"   Norm found: {norm_val:.2f}. Applying normalization...")
        
        class NormalizedPhysics(dinv.physics.Physics):
            """Wraps a physics operator to normalize by its operator norm."""
            def __init__(self, original_physics, norm):
                super().__init__()
                self.original = original_physics
                self.norm_const = norm
                
            def A(self, x):
                """Forward: A(x) / ||A||"""
                return self.original.A(x) / self.norm_const
            
            def A_adjoint(self, y):
                """Adjoint: A^T(y) / ||A||"""
                return self.original.A_adjoint(y) / self.norm_const
            
            def A_dagger(self, y):
                """Pseudo-inverse: A†(y) — filtered backprojection for CT.
                
                Much better initialization than A^T(y) for the inner solver.
                A† = (A^T A)^{-1} A^T gives the least-squares solution,
                while A^T just gives a blurry backprojection.
                """
                return self.original.A_dagger(y * self.norm_const)
                
            def forward(self, x): 
                return self.A(x)

        physics = NormalizedPhysics(physics, norm_val)
        return physics

    elif modality == "MRI":
        # --- MRI PHYSICS (Fourier Sampling) ---
        mask = torch.zeros((1, img_size, img_size))
        
        # Center k-space lines (always kept for low-frequency info)
        pad = (img_size - int(img_size * center_frac) + 1) // 2
        width = max(1, int(img_size * center_frac))
        mask[:, :, pad:pad + width] = 1.0
        
        # Random additional k-space lines for acceleration
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
    """
    The inner objective function h(w, θ) for the HOAG bilevel problem.
    
    This is the RECONSTRUCTION LOSS that the inner optimization minimizes.
    It corresponds to h_func_grad in the original HOAG:
        h(x, λ) = logistic_loss(x, X_train, y_train, exp(λ))  [original]
        h(w, θ) = ||y - Aw||² + exp(θ₀) · TV(w, exp(θ₁))     [ours]
    
    Components:
        1. Data Fidelity: ||y - Aw||²
           How well does the reconstruction w match the measured sinogram y
           through the forward model A (Radon transform)?
           
        2. Total Variation (TV) Regularization: exp(θ₀) · TV(w, exp(θ₁))
           Encourages piecewise-smooth images. TV penalizes the magnitude
           of image gradients, promoting sharp edges while suppressing noise.
           
           θ₀ controls HOW MUCH regularization (higher → smoother image)
           θ₁ controls the SMOOTHING of the TV norm (higher → smoother transition
              from penalized to unpenalized gradients, avoids non-differentiability)
    
    The exp(θ) parameterization ensures positivity and allows θ to range
    over all reals. This matches the original HOAG where λ is in log-space:
        np.exp(alpha[0])  # original hoag logistic.py line 28
    
    Args:
        w: Reconstructed image (shape: [B, C, H, W])
        theta: Hyperparameters [log(λ), log(ε)] 
        y: Measured sinogram/k-space data
        physics_op: Forward physics operator A
    
    Returns:
        Scalar loss value (data fidelity + weighted TV)
    """
    # --- DATA FIDELITY TERM: ||y - Aw||² ---
    # Measures how consistent the reconstruction is with the measurements
    fid = torch.norm(y - physics_op(w), p=2)**2
    
    # --- REGULARIZATION PARAMETERS ---
    # exp() ensures positivity; clamp prevents numerical overflow/underflow
    reg_weight = torch.exp(theta[0].clamp(max=1.0))   # λ = exp(θ₀)
    eps = torch.exp(theta[1].clamp(min=-12.0))         # ε = exp(θ₁)
    
    # --- TOTAL VARIATION (TV) REGULARIZATION ---
    # Compute image gradients using finite differences (circular shift)
    dx = torch.roll(w, 1, 2) - w   # Horizontal gradient
    dy = torch.roll(w, 1, 3) - w   # Vertical gradient
    
    # Isotropic TV: sqrt(dx² + dy² + ε)
    # The smoothing ε makes this differentiable (standard TV has a non-smooth |·| at 0)
    # This is the "smoothed Total Variation" commonly used in optimization
    tv_penalty = torch.mean(torch.sqrt(dx**2 + dy**2 + eps))
    
    return fid + reg_weight * tv_penalty


# ==========================================
#  3. NORMALIZATION UTILITY
# ==========================================
def robust_normalize(x):
    """
    Percentile-based normalization to [0, 1].
    
    Clips the top 1% of brightest pixels (streak artifacts from CT reconstruction)
    so they don't squash the useful signal into a narrow dynamic range.
    
    Used by both main.py and hoag.py for consistent input preprocessing.
    
    Args:
        x: Input tensor of shape (Batch, Channels, Height, Width)
    
    Returns:
        Normalized tensor in [0, 1] range
    """
    b = x.shape[0]
    x_flat = x.view(b, -1)
    
    # Calculate 1st and 99th percentiles per sample
    val_min = torch.quantile(x_flat, 0.01, dim=1).view(b, 1, 1, 1)
    val_max = torch.quantile(x_flat, 0.99, dim=1).view(b, 1, 1, 1)
    
    # Clip values to percentile range (removes extreme artifacts)
    x = torch.clamp(x, val_min, val_max)
    
    # Scale to [0, 1]
    denom = val_max - val_min
    denom = torch.where(denom > 1e-7, denom, torch.ones_like(denom))
    
    return (x - val_min) / denom