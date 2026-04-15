import torch
from torch.autograd import grad

# TODO: I don't think jacobian is needed here; torch.autograd.grad should be enough, to compute gradients of a scalar value function w.r.t. inputs

# batched jacobian
# y: [..., N], x: [..., M] -> [..., N, M]
def jacobian(y, x):
    ''' jacobian of y wrt x '''
    # Pre-allocate the Jacobian tensor
    jac = torch.zeros(*y.shape, x.shape[-1]).to(y.device)
    
    for i in range(y.shape[-1]):
        # 1. Flatten the specific output dimension
        y_flat = y[..., i].view(-1, 1)
        
        # 2. Compute gradient and capture it in a temporary variable
        grad_value = grad(y_flat, x, torch.ones_like(y_flat), 
                          create_graph=True, 
                          allow_unused=True)[0]
        
        # 3. Check for None and replace with zeros
        if grad_value is None:
            grad_value = torch.zeros_like(x)
            
        # 4. assign it to the pre-allocated tensor
        jac[..., i, :] = grad_value

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


