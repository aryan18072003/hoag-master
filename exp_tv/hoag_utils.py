import torch
import torch.autograd as autograd


# ==========================================
#  1. HESSIAN-VECTOR PRODUCT (HVP)
#     via Finite Differences
# ==========================================
def hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, v, fd_eps=1e-4):
    """
    Finite Difference HVP: H*v ~= (grad_h(w + eps*v) - grad_h(w - eps*v)) / (2*eps)
    
    This avoids 'second order derivative' errors in complex physics operators
    by strictly using first-order gradients.
    """
    # Ensure w_star and v are detached and ready
    w = w_star.detach()
    v = v.detach()
    
    # 1. Create Perturbed Inputs (w + eps*v) and (w - eps*v)
    w_plus = (w + fd_eps * v).requires_grad_(True)
    w_minus = (w - fd_eps * v).requires_grad_(True)
    
    # 2. Compute Gradient at w_plus
    with torch.enable_grad():
        loss_plus = inner_loss_fn(w_plus, theta.detach(), y, physics_op)
        grad_plus = torch.autograd.grad(loss_plus, w_plus, retain_graph=False)[0]
    
    # 3. Compute Gradient at w_minus
    with torch.enable_grad():
        loss_minus = inner_loss_fn(w_minus, theta.detach(), y, physics_op)
        grad_minus = torch.autograd.grad(loss_minus, w_minus, retain_graph=False)[0]
        
    # 4. Compute Finite Difference
    Hv = (grad_plus - grad_minus) / (2 * fd_eps)
    
    return Hv.detach()


# ==========================================
#  2. CONJUGATE GRADIENT (CG) SOLVER
# ==========================================
def conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, b,
                       max_iter=10, tol=1e-4, warm_start=None):

    # Bug #8 fix: Scale tolerance relative to ||b|| to avoid premature exit
    b_norm = b.detach().norm().item()
    scaled_tol = tol * max(b_norm, 1.0)

    if warm_start is not None and warm_start.shape == b.shape:
        x = warm_start.detach().clone()
    else:
        x = torch.zeros_like(b)
        warm_start = None
    
    # Initial residual: r = b - H*x
    if warm_start is not None:
        Ax = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, x)
        Ax = Ax + 1e-3 * x  # Tikhonov damping
        r = b.detach() - Ax
    else:
        r = b.detach().clone()
    
    p = r.clone()
    rsold = torch.sum(r * r)  # ||r||^2
    
    for i in range(max_iter):
        # Check convergence: ||r|| < scaled_tol
        if torch.sqrt(rsold) < scaled_tol:
            break
        
        # Compute H*p using finite-difference HVP
        Ap = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, p)
        
        Ap = Ap + 1e-3 * p  # Tikhonov damping matches above
        
        pAp = torch.sum(p * Ap)
        if pAp.abs() < 1e-12:
            break
        
        alpha = rsold / (pAp + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
    
    return x.detach()