import torch
import torch.autograd as autograd
from hoag_utils import hessian_vector_product, conjugate_gradient

def norm(img):
    img = torch.clamp(img, min=-150, max=250)
    img = (img + 150) / 400.0
    return img

def norm_z_score(img):
    mean = img.mean()
    std = img.std()
    
    if std > 0:
        img = (img - mean) / std
    else:
        img = torch.zeros_like(img)
        
    return img

# ==========================================
#  1. HOAG STATE
# ==========================================
class HOAGState:
    def __init__(self, epsilon_tol_init=1e-3, tolerance_decrease='exponential',
                 exponential_decrease_factor=0.9):

        self.epsilon_tol_init = epsilon_tol_init
        self.epsilon_tol = epsilon_tol_init
        self.tolerance_decrease = tolerance_decrease
        self.exponential_decrease_factor = exponential_decrease_factor
        
        self.q_warmstart = None
        
        self.g_func_old = float('inf')
        
        self.norm_init = None
        
        self.iteration = 0
        
        self.exact_epsilon = 1e-12
    
    def decrease_tolerance(self):
        self.iteration += 1
        if self.tolerance_decrease == 'quadratic':
            self.epsilon_tol = self.epsilon_tol_init / (self.iteration ** 2)
        elif self.tolerance_decrease == 'cubic':
            self.epsilon_tol = self.epsilon_tol_init / (self.iteration ** 3)
        elif self.tolerance_decrease == 'exponential':
            self.epsilon_tol *= self.exponential_decrease_factor
        else:
            raise NotImplementedError(f"Unknown schedule: {self.tolerance_decrease}")
        
        self.epsilon_tol = max(self.epsilon_tol, self.exact_epsilon)


# ==========================================
#  2. INNER PROBLEM SOLVER
# ==========================================
def solve_inner_problem(w_init, theta, y, physics_op, inner_loss_fn,
                        state, lr=0.02, max_steps=100, verbose=0):

    w = w_init.detach().clone()
    w.requires_grad_(True)
    
    optimizer = torch.optim.Adam([w], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.01)
    
    with torch.enable_grad():
        loss_init = inner_loss_fn(w, theta.detach(), y, physics_op)
        grad_init = autograd.grad(loss_init, w, retain_graph=False)[0]
        
        state.norm_init = grad_init.norm().item()
        
        if state.norm_init < 1e-12:
            state.norm_init = 1.0
    
    # Stopping threshold: epsilon_tol * norm_init
    stop_threshold = state.epsilon_tol * state.norm_init
    
    n_steps = 0
    for step in range(max_steps):
        optimizer.zero_grad()
        loss = inner_loss_fn(w, theta.detach(), y, physics_op)
        loss.backward()
        
        grad_norm = w.grad.norm().item()
        optimizer.step()
        scheduler.step()
        
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
#  3. HOAG STEP One iteration
# ==========================================
def hoag_step(theta, y, physics_op, model, loss_fn, mask,
              inner_loss_fn, state, inner_lr=0.02, inner_steps=100,
              cg_max_iter=20, verbose=0, modality="CT"):
    
    w_init = physics_op.A_dagger(y).detach().clone()
    
    w_star, n_inner = solve_inner_problem(
        w_init, theta, y, physics_op, inner_loss_fn,
        state, lr=inner_lr, max_steps=inner_steps, verbose=verbose
    )
    
    if verbose > 0:
        inner_loss_val = inner_loss_fn(w_star.detach(), theta.detach(), y, physics_op)
        print(f'  Inner objective: {inner_loss_val.item():.6f}')
    

    w_star = w_star.detach().requires_grad_(True) 
    if(modality=="CT"): x_in = norm(w_star)
    elif(modality=="MRI"): 
        magnitude = torch.sqrt(w_star[:, 0:1, :, :]**2 + w_star[:, 1:2, :, :]**2)
        x_in = norm_z_score(magnitude)
    pred = model(x_in)
    val_loss = loss_fn(pred, mask)
    val_loss_value = val_loss.item()
    
    val_loss_grad_w = autograd.grad(val_loss, w_star, retain_graph=True)[0]
    

    q = conjugate_gradient(
        inner_loss_fn, w_star, theta, y, physics_op,
        b=val_loss_grad_w,
        max_iter=cg_max_iter,
        tol=state.epsilon_tol,           
        warm_start=state.q_warmstart
    )
    
    state.q_warmstart = q.detach().clone()
    
    w_for_cross = w_star.detach().requires_grad_(True)
    theta_for_cross = theta.detach().requires_grad_(True)
    
    with torch.enable_grad():
        loss_cross = inner_loss_fn(w_for_cross, theta_for_cross, y, physics_op)
        grad_w_cross = autograd.grad(loss_cross, w_for_cross, create_graph=True)[0]
        grad_dot_q = torch.sum(grad_w_cross * q.detach())
        cross_grad_theta = autograd.grad(grad_dot_q, theta_for_cross, retain_graph=False)[0]
    
    implicit_term = -cross_grad_theta.detach()
    hyper_grad = implicit_term
    
    state.g_func_old = val_loss_value
    
    if verbose > 0:
        print(f'  Outer loss: {val_loss_value:.6f} | '
              f'||hyper_grad||: {hyper_grad.norm().item():.6f} | '
              f'epsilon_tol: {state.epsilon_tol:.2e}')
    
    return hyper_grad, val_loss_value, w_star
