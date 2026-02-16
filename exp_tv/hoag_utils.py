import torch
import torch.autograd as autograd


# ==========================================
#  1. HESSIAN-VECTOR PRODUCT (HVP)
#     via Finite Differences
# ==========================================
def hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, v, fd_eps=1e-4):

    v_detached = v.detach()
    
    # --- FORWARD PERTURBATION:  ---
    w_plus = (w_star.detach() + fd_eps * v_detached).requires_grad_(True)
    loss_plus = inner_loss_fn(w_plus, theta.detach(), y, physics_op)
    grad_plus = autograd.grad(loss_plus, w_plus, retain_graph=False)[0]
    
    # --- BACKWARD PERTURBATION:  ---
    w_minus = (w_star.detach() - fd_eps * v_detached).requires_grad_(True)
    loss_minus = inner_loss_fn(w_minus, theta.detach(), y, physics_op)
    grad_minus = autograd.grad(loss_minus, w_minus, retain_graph=False)[0]
    
    # --- CENTRAL DIFFERENCE:  ---
    Hv = (grad_plus - grad_minus) / (2.0 * fd_eps)
    
    return Hv.detach()


# ==========================================
#  2. CONJUGATE GRADIENT (CG) SOLVER
# ==========================================
def conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, b,
                       max_iter=10, tol=1e-4, warm_start=None):

    if warm_start is not None and warm_start.shape == b.shape:
        x = warm_start.detach().clone()
    else:
        x = torch.zeros_like(b)
        warm_start = None  
    
    # Initial residual: r = b - H·x
    if warm_start is not None:
        Ax = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, x)
        Ax = Ax + 1e-3 * x  
        r = b.detach() - Ax
    else:
        r = b.detach().clone()
    
    p = r.clone()             
    rsold = torch.sum(r * r)  # ||r||²
    
    for i in range(max_iter):
        # Check convergence: ||r|| < tol
        # This tolerance is tied to HOAG's epsilon_tol schedule
        if torch.sqrt(rsold) < tol:
            break
        
        # Compute H·p using finite-difference HVP
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