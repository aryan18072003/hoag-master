"""
hoag.py — PyTorch implementation of the HOAG Algorithm
=======================================================

HOAG = Hyperparameter Optimization with Approximate Gradient (Pedregosa, ICML 2016)

This module implements the HOAG bilevel optimization algorithm adapted for PyTorch,
following the structure of the original `hoag/hoag.py` (NumPy/SciPy version).

HOAG's Key Idea:
    Instead of grid search for hyperparameters, compute the GRADIENT of the 
    validation loss w.r.t. hyperparameters using implicit differentiation,
    then use gradient-based optimization (Adam) to update them.

The Bilevel Problem:
    Inner Level:  w*(θ) = argmin_w  h(w, θ)    [Training/reconstruction loss]
    Outer Level:  min_θ  g(w*(θ))               [Validation/task loss]

The Hypergradient (via Implicit Function Theorem):
    dg/dθ = ∂g/∂θ  -  (∂g/∂w) · H⁻¹ · (∂²h/∂w∂θ)
    where H = ∂²h/∂w² is the Hessian of the inner loss.

Reference: 
    Pedregosa, F. "Hyperparameter optimization with approximate gradient." 
    ICML 2016. http://jmlr.org/proceedings/papers/v48/pedregosa16.html
"""

import torch
import torch.autograd as autograd
from hoag_utils import hessian_vector_product, conjugate_gradient


# ==========================================
#  1. HOAG STATE — Tracks optimization state
#     across iterations (like Bxk, epsilon_tol
#     in the original hoag_lbfgs function)
# ==========================================
class HOAGState:
    """
    Maintains persistent state across HOAG iterations.
    
    In the original HOAG (hoag.py lines 82-98), several variables persist
    across outer iterations. This class packages them together.
    
    Attributes:
        epsilon_tol (float): Current tolerance for inner solver and CG.
            Starts at epsilon_tol_init (1e-3) and decreases each iteration.
            This is the KEY mechanism from HOAG — by solving the inner problem
            with increasing precision, we save computation in early iterations
            while still converging to the correct hypergradient.
            
        q_warmstart (Tensor or None): Warm-start for the CG solver.
            Corresponds to `Bxk` in the original (hoag.py line 88).
            Reusing the previous CG solution as the starting point
            dramatically speeds up convergence since consecutive solutions
            are usually similar.
            
        g_func_old (float): Previous outer loss value.
            Used for monitoring convergence (original hoag.py line 90).
            
        norm_init (float or None): Initial gradient norm for stopping criterion.
            The inner solver stops when ||∇h|| < epsilon_tol * norm_init
            (original hoag.py lines 97, 118-121).
            
        tolerance_decrease (str): Schedule for decreasing epsilon_tol.
            Options: 'exponential' (×0.9), 'quadratic' (1/k²), 'cubic' (1/k³)
            (original hoag.py lines 162-171).
            
        iteration (int): Current outer iteration counter.
    """
    
    def __init__(self, epsilon_tol_init=1e-3, tolerance_decrease='exponential',
                 exponential_decrease_factor=0.9):
        """
        Args:
            epsilon_tol_init: Starting tolerance (original default: 1e-3)
            tolerance_decrease: How tolerance shrinks ('exponential', 'quadratic', 'cubic')
            exponential_decrease_factor: Multiplier per iteration for exponential schedule
                                         (original default: 0.9, see hoag.py line 14)
        """
        self.epsilon_tol_init = epsilon_tol_init
        self.epsilon_tol = epsilon_tol_init
        self.tolerance_decrease = tolerance_decrease
        self.exponential_decrease_factor = exponential_decrease_factor
        
        # CG warm-start (corresponds to Bxk in original hoag.py line 88)
        self.q_warmstart = None
        
        # Previous outer loss for monitoring (original hoag.py line 90)
        self.g_func_old = float('inf')
        
        # Initial gradient norm for inner stopping criterion (original hoag.py line 97)
        self.norm_init = None
        
        # Iteration counter for quadratic/cubic schedules
        self.iteration = 0
        
        # Minimum tolerance floor (original hoag.py line 82, exact_epsilon = 1e-12)
        self.exact_epsilon = 1e-12
    
    def decrease_tolerance(self):
        """
        Decrease the tolerance after each outer iteration.
        
        This mirrors the original hoag.py lines 162-173:
            if tolerance_decrease == 'quadratic':
                epsilon_tol = epsilon_tol_init / (it ** 2)
            elif tolerance_decrease == 'exponential':
                epsilon_tol *= exponential_decrease_factor
        
        The key insight: early iterations use loose tolerance (fast but rough),
        later iterations use tight tolerance (slow but precise). This gives
        HOAG its computational advantage over exact methods.
        """
        self.iteration += 1
        
        if self.tolerance_decrease == 'quadratic':
            # Original: epsilon_tol_init / (it ** 2) — see hoag.py line 163
            self.epsilon_tol = self.epsilon_tol_init / (self.iteration ** 2)
        elif self.tolerance_decrease == 'cubic':
            # Original: epsilon_tol_init / (it ** 3) — see hoag.py line 165
            self.epsilon_tol = self.epsilon_tol_init / (self.iteration ** 3)
        elif self.tolerance_decrease == 'exponential':
            # Original: epsilon_tol *= factor — see hoag.py line 167
            self.epsilon_tol *= self.exponential_decrease_factor
        else:
            raise NotImplementedError(f"Unknown schedule: {self.tolerance_decrease}")
        
        # Floor the tolerance (original hoag.py line 173)
        self.epsilon_tol = max(self.epsilon_tol, self.exact_epsilon)


# ==========================================
#  2. INNER PROBLEM SOLVER
#     Corresponds to the L-BFGS-B inner loop
#     in hoag.py lines 101-138
# ==========================================
def solve_inner_problem(w_init, theta, y, physics_op, inner_loss_fn,
                        state, lr=0.02, max_steps=100, verbose=0):
    """
    Solve the inner optimization problem: w* = argmin_w h(w, θ)
    
    This corresponds to the L-BFGS-B inner loop in the original HOAG
    (hoag.py lines 101-138). We use Adam instead of L-BFGS-B since we're
    in PyTorch, but the KEY HOAG feature is preserved: the stopping criterion
    based on the current tolerance epsilon_tol.
    
    Original stopping criterion (hoag.py lines 118-121):
        ||∇h(w)|| < epsilon_tol * norm_init * exp(min(λ) - min(λ₀))
    
    We simplify to:
        ||∇h(w)|| < epsilon_tol * norm_init
    
    This means early outer iterations solve the inner problem roughly (fast),
    and later iterations solve it more precisely (accurate hypergradient).
    
    Args:
        w_init: Initial reconstruction (typically A^T y)
        theta: Current hyperparameters [log_lambda, log_epsilon]
        y: Measurement data (sinogram for CT)
        physics_op: Forward physics operator
        inner_loss_fn: h(w, θ, y, A) — the inner objective
        state: HOAGState with current tolerance
        lr: Learning rate for Adam inner solver
        max_steps: Maximum inner iterations
        verbose: Print level
    
    Returns:
        w_star: Approximate minimizer of the inner problem
        n_steps: Number of inner steps taken
    """
    # Initialize w from the provided starting point
    w = w_init.detach().clone()
    w.requires_grad_(True)
    
    # Use Adam for the inner solve (better than vanilla GD for this problem)
    optimizer = torch.optim.Adam([w], lr=lr)
    
    # Compute initial gradient norm for the stopping criterion
    # This corresponds to norm_init in the original (hoag.py line 97)
    with torch.enable_grad():
        loss_init = inner_loss_fn(w, theta.detach(), y, physics_op)
        grad_init = autograd.grad(loss_init, w, retain_graph=False)[0]
        
        if state.norm_init is None:
            # First call — set the reference gradient norm
            state.norm_init = grad_init.norm().item()
        
        # Avoid division by zero
        if state.norm_init < 1e-12:
            state.norm_init = 1.0
    
    # Stopping threshold: epsilon_tol * norm_init
    # (simplified from original hoag.py line 118-119)
    stop_threshold = state.epsilon_tol * state.norm_init
    
    n_steps = 0
    for step in range(max_steps):
        optimizer.zero_grad()
        loss = inner_loss_fn(w, theta.detach(), y, physics_op)
        loss.backward()
        
        # Check HOAG stopping criterion: ||∇h|| < epsilon_tol * norm_init
        # (original hoag.py lines 118-121)
        grad_norm = w.grad.norm().item()
        
        optimizer.step()
        
        # Non-negativity constraint for image reconstruction
        with torch.no_grad():
            w.clamp_(0.0, 1.0)
        
        n_steps += 1
        
        # HOAG-style early stopping based on tolerance
        if grad_norm < stop_threshold:
            if verbose > 1:
                print(f'    Inner converged at step {n_steps}, '
                      f'grad_norm={grad_norm:.6f} < threshold={stop_threshold:.6f}')
            break
    
    if verbose > 0:
        print(f'  Inner: {n_steps} steps, final grad_norm={grad_norm:.6f}, '
              f'threshold={stop_threshold:.6f}')
    
    return w, n_steps


# ==========================================
#  3. HOAG STEP — One complete outer iteration
#     This is the core of the HOAG algorithm,
#     combining all pieces together.
#     Corresponds to one iteration of the outer
#     loop in hoag.py lines 101-240.
# ==========================================
def hoag_step(theta, y, physics_op, model, loss_fn, mask,
              inner_loss_fn, state, inner_lr=0.02, inner_steps=100,
              cg_max_iter=20, verbose=0):
    """
    Perform one complete HOAG outer iteration.
    
    This function follows the structure of one iteration from the original 
    hoag_lbfgs() (hoag.py lines 101-240), adapted for the CT reconstruction
    + segmentation pipeline.
    
    Algorithm Steps (mapping to original hoag.py):
    
    Step 1: Solve inner problem approximately           [hoag.py lines 101-138]
             w* = argmin_w h(w, θ)
             Stopping: ||∇h|| < ε_tol * ||∇h₀||
    
    Step 2: Compute Hessian of inner loss               [hoag.py lines 147-150]
             H = ∂²h/∂w²  (as a linear operator via autograd)
    
    Step 3: Compute outer loss gradient ∂g/∂w           [hoag.py line 152]
             where g = DiceBCE(UNet(w*), mask)
    
    Step 4: Solve linear system H·q = ∂g/∂w via CG     [hoag.py line 158]
             with warm-start from previous iteration
    
    Step 5: Compute hypergradient via implicit diff     [hoag.py line 175]
             ∇θ = ∂g/∂θ - (∂²h/∂w∂θ)ᵀ · q
    
    Step 6: Decrease tolerance for next iteration       [hoag.py lines 162-173]
    
    Args:
        theta: Current hyperparameters (requires_grad=True)
        y: Noisy measurement data (sinogram)
        physics_op: Forward physics operator (Radon transform)
        model: The U-Net segmentation network
        loss_fn: Outer loss function (DiceBCE)
        mask: Ground truth segmentation mask
        inner_loss_fn: Inner loss function h(w, θ, y, A)
        state: HOAGState tracking tolerance, warm-start, etc.
        inner_lr: Learning rate for inner Adam solver
        inner_steps: Max inner optimization steps
        cg_max_iter: Max CG iterations for linear system solve
        verbose: Print level (0=silent, 1=summary, 2=detailed)
    
    Returns:
        hyper_grad: The HOAG hypergradient ∂g/∂θ (to be used by Adam)
        val_loss_value: Scalar value of the outer loss
        w_star: The inner problem solution (for U-Net training in Phase 4)
    """
    
    # ------------------------------------------------------------------
    # STEP 1: Solve Inner Problem (original hoag.py lines 101-138)
    # ------------------------------------------------------------------
    # Initialize w from the adjoint (backprojection) — standard starting point
    w_init = physics_op.A_dagger(y).detach().clone()
    
    # Solve: w* = argmin_w  ||y - Aw||² + exp(θ₀)·TV(w, exp(θ₁))
    # Uses tolerance-based stopping from HOAG
    w_star, n_inner = solve_inner_problem(
        w_init, theta, y, physics_op, inner_loss_fn,
        state, lr=inner_lr, max_steps=inner_steps, verbose=verbose
    )
    
    if verbose > 0:
        inner_loss_val = inner_loss_fn(w_star.detach(), theta.detach(), y, physics_op)
        print(f'  Inner objective: {inner_loss_val.item():.6f}')
    
    # ------------------------------------------------------------------
    # STEP 2: Compute Hessian of Inner Loss (original hoag.py lines 147-150)
    # ------------------------------------------------------------------
    # In the original: fhs = h_hessian(x, lambdak)
    #                  B_op = LinearOperator(matvec=lambda z: fhs(z))
    # In PyTorch: we use FINITE-DIFFERENCE HVP (hessian_vector_product)
    # The Hessian H = ∂²h/∂w² is NOT explicitly formed — only H·v products
    # are computed via finite differences to avoid grid_sampler issues.
    
    # ------------------------------------------------------------------
    # STEP 3: Compute Outer Loss & Its Gradient w.r.t. w
    # (original hoag.py line 152: g_func, g_grad = g_func_grad(x, lambdak))
    # ------------------------------------------------------------------
    # Detach w_star from inner optimization graph, re-enable gradient tracking
    w_star = w_star.detach().requires_grad_(True)
    
    # Forward through U-Net: normalize → segment → compare with ground truth
    # This is the outer/validation loss g(w*(θ))
    from physics import robust_normalize  # Import from physics to avoid circular imports
    x_mag = torch.sqrt(w_star[:, 0:1]**2 + w_star[:, 1:2]**2 + 1e-8)
    x_in = robust_normalize(x_mag)
    
    pred = model(x_in)
    val_loss = loss_fn(pred, mask)
    val_loss_value = val_loss.item()
    
    # g_grad = ∂g/∂w  (gradient of outer loss w.r.t. reconstruction w*)
    # This is the "b" vector in the CG system H·q = b
    val_loss_grad_w = autograd.grad(val_loss, w_star, retain_graph=True)[0]
    
    # ------------------------------------------------------------------
    # STEP 4: Solve Linear System H·q = ∂g/∂w via CG
    # (original hoag.py line 158: Bxk, success = cg(B_op, g_grad, x0=Bxk, tol=tol_CG))
    # ------------------------------------------------------------------
    # This is the most computationally expensive step.
    # We solve: (∂²h/∂w²) · q = ∂g/∂w
    # 
    # The solution q tells us how a small change in w (driven by the outer loss)
    # would need to be "inverted" through the curvature of the inner problem.
    #
    # CG warm-start: reuse the solution from the previous outer iteration
    # (original hoag.py line 153: if Bxk is None: Bxk = x.copy())
    q = conjugate_gradient(
        inner_loss_fn, w_star, theta, y, physics_op,
        b=val_loss_grad_w,
        max_iter=cg_max_iter,
        tol=state.epsilon_tol,           # CG tolerance matches HOAG tolerance schedule
        warm_start=state.q_warmstart     # Warm-start from previous iteration
    )
    
    # Save for next iteration's warm-start (original hoag.py line 158)
    state.q_warmstart = q.detach().clone()
    
    # ------------------------------------------------------------------
    # STEP 5: Compute Hypergradient via Implicit Differentiation
    # (original hoag.py line 175: grad_lambda = -h_crossed(x, lambdak).dot(Bxk))
    # ------------------------------------------------------------------
    # The full hypergradient is:
    #   dg/dθ = ∂g/∂θ  -  (∂²h/∂w∂θ)ᵀ · q
    #            ↑              ↑
    #       direct term    implicit term (from Implicit Function Theorem)
    #
    # IMPLICIT TERM via Finite Differences:
    #   Exact autograd fails because inner_loss_fn uses physics_op (grid_sample).
    #   For each hyperparameter θ_j:
    #     (∂²h/∂w∂θ_j)ᵀ · q ≈ [∇_w h(w*, θ+ε·eⱼ) - ∇_w h(w*, θ-ε·eⱼ)]ᵀ · q / (2ε)
    
    w_detached = w_star.detach().requires_grad_(True)
    q_detached = q.detach()
    fd_eps = 1e-4
    
    implicit_term = torch.zeros_like(theta)
    
    for j in range(theta.shape[0]):
        e_j = torch.zeros_like(theta)
        e_j[j] = fd_eps
        
        # ∇_w h(w*, θ + ε·eⱼ)
        theta_plus = (theta.detach() + e_j).requires_grad_(False)
        w_p = w_detached.detach().requires_grad_(True)
        loss_plus = inner_loss_fn(w_p, theta_plus, y, physics_op)
        grad_plus = autograd.grad(loss_plus, w_p, retain_graph=False)[0]
        
        # ∇_w h(w*, θ - ε·eⱼ)
        theta_minus = (theta.detach() - e_j).requires_grad_(False)
        w_m = w_detached.detach().requires_grad_(True)
        loss_minus = inner_loss_fn(w_m, theta_minus, y, physics_op)
        grad_minus = autograd.grad(loss_minus, w_m, retain_graph=False)[0]
        
        # (∂²h/∂w∂θ_j)ᵀ · q via finite difference
        cross_deriv_dot_q = torch.sum((grad_plus - grad_minus) / (2.0 * fd_eps) * q_detached)
        implicit_term[j] = -cross_deriv_dot_q  # Negative from IFT
    
    # DIRECT TERM: ∂g/∂θ (zero — outer loss doesn't depend on θ directly)
    direct_grad = torch.zeros_like(theta)
    
    # FINAL HYPERGRADIENT
    hyper_grad = direct_grad + implicit_term
    
    # ------------------------------------------------------------------
    # COMMENTED OUT: Exact Autograd Cross-Derivative
    # Does NOT work — inner_loss_fn uses physics_op (grid_sample → no 2nd derivatives)
    # ------------------------------------------------------------------
    # w_for_cross = w_star.detach().requires_grad_(True)
    # theta_for_cross = theta.detach().requires_grad_(True)
    # with torch.enable_grad():
    #     loss_cross = inner_loss_fn(w_for_cross, theta_for_cross, y, physics_op)
    #     grad_w_cross = autograd.grad(loss_cross, w_for_cross, create_graph=True)[0]
    #     grad_dot_q = torch.sum(grad_w_cross * q.detach())
    #     cross_grad_theta = autograd.grad(grad_dot_q, theta_for_cross, retain_graph=False)[0]
    # implicit_term = -cross_grad_theta.detach()
    # direct_grad = torch.zeros_like(theta)
    # hyper_grad = direct_grad + implicit_term
    
    # ------------------------------------------------------------------
    # STEP 6: Decrease Tolerance (original hoag.py lines 162-173)
    # ------------------------------------------------------------------
    # The tolerance schedule is the key to HOAG's efficiency:
    # - Early iterations: loose tolerance → fast but rough hypergradient
    # - Later iterations: tight tolerance → slow but precise hypergradient
    # This gives near-exact convergence with much less computation than
    # always solving the inner problem exactly.
    state.g_func_old = val_loss_value
    state.decrease_tolerance()
    
    if verbose > 0:
        print(f'  Outer loss: {val_loss_value:.6f} | '
              f'||hyper_grad||: {hyper_grad.norm().item():.6f} | '
              f'epsilon_tol: {state.epsilon_tol:.2e}')
    
    return hyper_grad, val_loss_value, w_star
