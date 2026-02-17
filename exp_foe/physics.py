import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepinv as dinv


# ==========================================
#  FoE CONFIGURATION
# ==========================================
NUM_EXPERTS = 5       # J: number of expert filters (paper: 10)
FILTER_SIZE = 5       # K: spatial filter size (paper: 7)
IN_CHANNELS = 1      # C: input channels 

N_SCALAR_PARAMS = 1 + NUM_EXPERTS + NUM_EXPERTS   # 1 + J + J = 11
N_FILTER_PARAMS = NUM_EXPERTS * IN_CHANNELS * FILTER_SIZE * FILTER_SIZE  # 5*1*25 = 125
THETA_SIZE = N_SCALAR_PARAMS + N_FILTER_PARAMS     # 11 + 125 = 136


# ==========================================
#  1. PHYSICS OPERATOR FACTORY
# ==========================================
def get_physics_operator(img_size, acceleration, center_frac, device, modality="CT"):
    """
    Creates a differentiable physics forward model for CT or MRI.
    Identical to my_exp/physics.py — physics doesn't change with regularizer.
    """
    if modality == "CT":
        if acceleration == 1:
            num_views = 180
        else:
            num_views = int(180 / acceleration)

        angles = torch.linspace(0, 180, num_views).to(device)

        physics = dinv.physics.Tomography(
            angles=angles,
            img_width=img_size,
            circle=False,
            device=device,
            normalize=True    
        )

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
#  2. theta PARSING
# ==========================================
def parse_theta(theta):

    idx = 0
    global_weight = theta[idx]
    idx += 1

    filter_weights = theta[idx : idx + NUM_EXPERTS]
    idx += NUM_EXPERTS

    smoothing_params = theta[idx : idx + NUM_EXPERTS]
    idx += NUM_EXPERTS

    filters = theta[idx:].view(NUM_EXPERTS, IN_CHANNELS, FILTER_SIZE, FILTER_SIZE)
    return global_weight, filter_weights, smoothing_params, filters


# ==========================================
#  3. θ INITIALIZATION
# ==========================================
def initialize_theta(device):
    """
    Create initial θ for the paper's FoE regularizer.

    Filters initialized with derivative-like 5×5 kernels applied to
    both channels identically (real + imaginary have similar structure).

    Returns:
        theta: Flat tensor of size THETA_SIZE on the given device
    """
    # --- Global weight: e^{0} = 1.0 ---
    global_weight = torch.tensor([0.0])

    # --- Per-filter weights: e^{-2} ≈ 0.135 each ---
    filter_weights = torch.full((NUM_EXPERTS,), -2.0)

    # --- Smoothing params: ν_j = exp(-4.6) ≈ 0.01 (small → sharp L1 approx) ---
    smoothing_params = torch.full((NUM_EXPERTS,), -4.6)

    # --- Expert Filters (J × C × K × K) ---
    # Initialize with edge/texture detectors, same pattern in both channels
    filters = torch.zeros(NUM_EXPERTS, IN_CHANNELS, FILTER_SIZE, FILTER_SIZE)

    # Helper: set same spatial pattern in both channels (scaled by 1/√C)
    scale = 1.0 / np.sqrt(IN_CHANNELS)

    # Filter 1: Horizontal gradient (Prewitt-like 5×5)
    h_pattern = torch.tensor([
        [-1, -1,  0,  1,  1],
        [-2, -2,  0,  2,  2],
        [-3, -3,  0,  3,  3],
        [-2, -2,  0,  2,  2],
        [-1, -1,  0,  1,  1]
    ], dtype=torch.float32) / 12.0
    for c in range(IN_CHANNELS):
        filters[0, c] = h_pattern * scale

    # Filter 2: Vertical gradient
    for c in range(IN_CHANNELS):
        filters[1, c] = h_pattern.T * scale

    # Filter 3: Diagonal (45°)
    d_pattern = torch.tensor([
        [-2, -1,  0,  0,  0],
        [-1, -2,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  2,  1],
        [ 0,  0,  0,  1,  2]
    ], dtype=torch.float32) / 6.0
    for c in range(IN_CHANNELS):
        filters[2, c] = d_pattern * scale

    # Filter 4: Anti-diagonal (135°)
    for c in range(IN_CHANNELS):
        filters[3, c] = d_pattern.flip(1) * scale

    # Filter 5: Laplacian of Gaussian (blob/texture)
    log_pattern = torch.tensor([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0]
    ], dtype=torch.float32) / 16.0
    for c in range(IN_CHANNELS):
        filters[4, c] = log_pattern * scale

    # Flatten into θ vector
    theta = torch.cat([global_weight, filter_weights, smoothing_params, filters.flatten()])
    return theta.to(device)


# ==========================================
#  4. INNER LOSS (h in HOAG) 
# ==========================================
def inner_loss_func(w, theta, y, physics_op):

    residual = y - physics_op(w)
    fid = torch.mean(residual ** 2)

    # --- PARSE θ ---
    global_weight, filter_weights, smoothing_params, filters = parse_theta(theta)

    # --- FIELD OF EXPERTS REGULARIZATION  ---
    #
    # R_θ(w) = e^{θ} · Σⱼ e^{θⱼ} · ||cⱼ * w||_{ν_j}
    #
    # where ||x||_ν = mean(√(xᵢ² + ν²) - ν)   [smoothed 1-norm]
    #
    foe_sum = torch.tensor(0.0, device=w.device)

    for j in range(NUM_EXPERTS):
        # Convolution: cⱼ * w  (multi-channel: cⱼ has shape (1, C, K, K))
        c_j = filters[j:j+1]  # shape: (1, C, K, K)
        response = F.conv2d(w, c_j, padding=FILTER_SIZE // 2)  # (B, 1, H, W)

        # Smoothing parameter: ν_j = exp(θ_{J+j}) — positive by construction
        nu_j = torch.exp(smoothing_params[j].clamp(max=2.0))

        # Smoothed 1-norm: ||cⱼ * w||_{ν_j} = mean(√(z^2 + ν^2) - ν)
        smoothed_norm = torch.mean(torch.sqrt(response ** 2 + nu_j ** 2) - nu_j)

        # Per-filter weight: e^{θⱼ}
        foe_sum = foe_sum + torch.exp(filter_weights[j].clamp(max=4.0)) * smoothed_norm

    # Global weight: e^{θ₀}
    foe_reg = torch.exp(global_weight.clamp(max=4.0)) * foe_sum

    return fid + foe_reg


# ==========================================
#  5. NORMALIZATION UTILITY
# ==========================================
def robust_normalize(x):
    """Fixed-range normalization to [0, 1].
    
    MSDDataset already windows CT data to [0, 1] (clip to [-150, 250]
    then scale). The true signal is always in [0, 1] — FBP reconstruction
    only adds noise/artifacts OUTSIDE this range.
    
    Clamping to [0, 1] removes noise while preserving the signal structure,
    so both clean images and FBP reconstructions have the same distribution
    (mean ~0.14, same spatial structure). This prevents the domain shift
    that adaptive percentile normalization causes (FBP mean 0.14 -> 0.51).
    """
    return x.clamp(0.0, 1.0)