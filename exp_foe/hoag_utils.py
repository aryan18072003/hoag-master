"""
hoag_utils.py — Mathematical Utility Functions for HOAG
=======================================================

This module provides the core mathematical operations used by hoag.py:
1. Hessian-Vector Products (HVP) via FINITE DIFFERENCES
2. Conjugate Gradient (CG) solver for the linear system H·q = b

These correspond to the inner subroutines called by the HOAG algorithm.
In the original NumPy implementation (hoag/hoag.py), these were handled by:
    - scipy.sparse.linalg.LinearOperator (for Hessian-vector products)
    - scipy.sparse.linalg.cg (for the CG solver)

HVP IMPLEMENTATION:
    Uses finite-difference approximation for the Hessian-vector product.
    
    Exact autograd (create_graph=True) does NOT work here because DeepInv's
    Tomography operator uses grid_sample, which has no second derivative
    implementation (RuntimeError: derivative for aten::cudnn_grid_sampler_backward
    is not implemented).
    
    FD formula: H·v ≈ (∇h(w + εv) - ∇h(w - εv)) / (2ε)
    Only requires first-order gradients — works with any operator.
"""

import torch
import torch.autograd as autograd


# ==========================================
#  1. HESSIAN-VECTOR PRODUCT (HVP)
#     via Finite Differences
# ==========================================
def hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, v, fd_eps=1e-4):
    """
    Computes the Hessian-vector product H·v using central finite differences.
    
    This corresponds to the Hessian operator in the original HOAG:
        fhs = h_hessian(x, lambdak)         # hoag.py line 147
        B_op = LinearOperator(matvec=fhs)    # hoag.py line 148-150
    
    Formula: H·v ≈ (∇h(w + ε·v) - ∇h(w - ε·v)) / (2ε)
    
    This is O(ε²) accurate and only requires first-order gradients.
    
    Args:
        inner_loss_fn: Inner loss function h(w, θ, y, A)
        w_star: Current inner solution w*
        theta: Current hyperparameters
        y: Measurement data
        physics_op: Forward physics operator
        v: Direction vector to multiply with the Hessian
        fd_eps: Finite difference step size (default: 1e-4)
    
    Returns:
        Hv: The product H·v (same shape as w_star)
    """
    v_detached = v.detach()
    
    # ∇h(w + ε·v)
    w_plus = (w_star.detach() + fd_eps * v_detached).requires_grad_(True)
    loss_plus = inner_loss_fn(w_plus, theta.detach(), y, physics_op)
    grad_plus = autograd.grad(loss_plus, w_plus, retain_graph=False)[0]
    
    # ∇h(w - ε·v)
    w_minus = (w_star.detach() - fd_eps * v_detached).requires_grad_(True)
    loss_minus = inner_loss_fn(w_minus, theta.detach(), y, physics_op)
    grad_minus = autograd.grad(loss_minus, w_minus, retain_graph=False)[0]
    
    # Central difference: H·v ≈ (∇⁺ - ∇⁻) / (2ε)
    Hv = (grad_plus - grad_minus) / (2.0 * fd_eps)
    
    return Hv.detach()


# ==========================================
#  COMMENTED OUT: Exact Autograd HVP
#  Does NOT work with DeepInv Tomography (grid_sample has no 2nd derivatives)
# ==========================================
# def hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, v, fd_eps=1e-4):
#     """Exact HVP via double backprop: H·v = ∇_w(⟨∇_w h, v⟩)"""
#     w = w_star.detach().requires_grad_(True)
#     with torch.enable_grad():
#         loss = inner_loss_fn(w, theta.detach(), y, physics_op)
#         grad_w = autograd.grad(loss, w, create_graph=True)[0]
#         grad_dot_v = torch.sum(grad_w * v.detach())
#         Hv = autograd.grad(grad_dot_v, w, retain_graph=False)[0]
#     return Hv.detach()


# ==========================================
#  2. CONJUGATE GRADIENT (CG) SOLVER
# ==========================================
def conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, b,
                       max_iter=10, tol=1e-4, warm_start=None):
    """
    Solves the linear system H·x = b using the Conjugate Gradient method.
    
    This corresponds to the CG solver call in the original HOAG:
        Bxk, success = splinalg.cg(B_op, g_grad, x0=Bxk, tol=tol_CG)
        # hoag.py line 158
    
    Why CG instead of direct solve?
        The Hessian H is never explicitly formed (it's too large).
        CG only needs matrix-vector products H·v, which we compute via
        finite differences (see hessian_vector_product above).
        CG converges in at most N iterations for an N×N positive-definite
        system, but often converges much faster.
    
    HOAG-specific features:
        - tol: Tied to HOAG's tolerance schedule (epsilon_tol)
          Early iterations: loose tolerance → fast CG (few iterations)
          Later iterations: tight tolerance → accurate CG (more iterations)
        - warm_start: Reuses previous iteration's solution (Bxk in original)
          Consecutive HOAG iterations have similar solutions, so warm-starting
          dramatically reduces CG iterations needed.
    
    Args:
        inner_loss_fn: Inner loss function (used for HVP computation)
        w_star: Current inner solution
        theta: Current hyperparameters
        y: Measurement data
        physics_op: Forward physics operator
        b: Right-hand side vector (typically ∂g/∂w — the outer loss gradient)
        max_iter: Maximum CG iterations
        tol: Convergence tolerance (from HOAG's epsilon_tol schedule)
        warm_start: Previous solution for initialization (from state.q_warmstart)
    
    Returns:
        x: Approximate solution to H·x = b
    """
    # Initialize x — use warm-start if available (original hoag.py line 153-154)
    # Warm-starting from the previous HOAG iteration speeds up CG significantly
    # NOTE: Discard warm-start if shapes don't match (last batch may be smaller)
    if warm_start is not None and warm_start.shape == b.shape:
        x = warm_start.detach().clone()
    else:
        x = torch.zeros_like(b)
        warm_start = None  # Reset so we skip the residual computation below
    
    # Initial residual: r = b - H·x
    if warm_start is not None:
        Ax = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, x)
        Ax = Ax + 1e-3 * x  # Tikhonov damping
        r = b.detach() - Ax
    else:
        r = b.detach().clone()
    
    p = r.clone()             # Search direction
    rsold = torch.sum(r * r)  # ||r||²
    
    for i in range(max_iter):
        # Check convergence: ||r|| < tol
        # This tolerance is tied to HOAG's epsilon_tol schedule
        if torch.sqrt(rsold) < tol:
            break
        
        # Compute H·p using finite-difference HVP
        Ap = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, p)
        
        # TIKHONOV DAMPING: Add small regularization (1e-3 * p) to H·p
        # This ensures positive-definiteness even if H is nearly singular.
        # Without this, CG can diverge on ill-conditioned problems.
        # (The original HOAG doesn't need this because logistic regression
        #  Hessians are always positive-definite, but TV-regularized CT
        #  reconstruction may have near-singular Hessians)
        Ap = Ap + 1e-3 * p
        
        # Standard CG update formulas:
        #   α = rᵀr / pᵀAp          (step size along search direction)
        #   x = x + α·p             (update solution)
        #   r = r - α·Ap            (update residual)
        #   β = rᵀr_new / rᵀr_old  (conjugate direction coefficient)
        #   p = r + β·p             (new search direction)
        pAp = torch.sum(p * Ap)
        if pAp.abs() < 1e-12:
            # Degenerate direction — H·p ≈ 0, skip this iteration
            break
        
        alpha = rsold / (pAp + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
    
    return x.detach()