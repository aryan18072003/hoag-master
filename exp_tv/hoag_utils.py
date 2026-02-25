import torch
import torch.autograd as autograd


# ==========================================
#  1. HESSIAN-VECTOR PRODUCT (HVP)
#     via Exact Decomposition
# ==========================================
def hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, v, fd_eps=None):

    v_detached = v.detach()
    
    # --- Term 1: Data fidelity Hessian-vector product ---
    # H_fid = (2/n) * AᵀA  where n = y.numel()
    with torch.no_grad():
        Av = physics_op.A(v_detached)
        AtAv = physics_op.A_adjoint(Av)
    n = y.numel()
    term1 = (2.0 / n) * AtAv
    
    # --- Term 2: Regularizer Hessian-vector product ---
    from physics import regularizer_only
    w_for_reg = w_star.detach().requires_grad_(True)
    with torch.enable_grad():
        reg = regularizer_only(w_for_reg, theta.detach())
        grad_reg = autograd.grad(reg, w_for_reg, create_graph=True)[0]
        grad_dot_v = torch.sum(grad_reg * v_detached)
        term2 = autograd.grad(grad_dot_v, w_for_reg, retain_graph=False)[0]
    
    return (term1 + term2.detach()).detach()


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
        Ax = Ax + 1e-3 * x 
        r = b.detach() - Ax
    else:
        r = b.detach().clone()
    
    p = r.clone()
    rsold = torch.sum(r * r)  # ||r||^2
    
    for i in range(max_iter):
        # Check convergence: ||r|| < scaled_tol
        if torch.sqrt(rsold) < scaled_tol:
            break

        Ap = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, p)
        
        Ap = Ap + 1e-3 * p 
        
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