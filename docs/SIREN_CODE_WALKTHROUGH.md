# SIREN in DeepReach: Code Walkthrough & Practical Guide

## Complete End-to-End Flow: From Input to Safe Control

This document traces exactly how SIREN networks enable smooth, differentiable safety gradients in your DeepReach implementation.

---

## Step 1: Network Initialization (Setup Phase)

### File: `utils/modules.py`

#### Creating the Network

From lines 118-125:
```python
class SingleBVPNet(nn.Module):
    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.net = FCBlock(in_features=in_features, out_features=out_features, 
                          num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, 
                          outermost_linear=True, 
                          nonlinearity=type)  # type='sine' for SIREN
```

**Key parameters:**
- `type='sine'`: Selects SIREN activation
- `num_hidden_layers=3`: Depth of network (3 layers with sin activation)
- `hidden_features=256`: Width of each hidden layer
- `outermost_linear=True`: Output layer has **no activation** (just linear)

#### Building the FCBlock with SIREN

From lines 57-101:
```python
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', ...):
        
        # SIREN configuration
        nls_and_inits = {
            'sine': (Sine(), sine_init, first_layer_sine_init),  # ← Our path
            'relu': (nn.ReLU(), init_weights_normal, None),
            # ...
        }
        
        nl, nl_weight_init, first_layer_init = nls_and_inits['sine']
        
        # Build network with sin activations
        self.net = []
        
        # First layer: Linear → Sine (with special init)
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), 
            nl  # This is Sine()
        ))
        
        # Hidden layers: Linear → Sine
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), 
                nl  # sin(30·)
            ))
        
        # Output layer: Linear only (outermost_linear=True)
        self.net.append(nn.Sequential(
            BatchLinear(hidden_features, out_features)
        ))
        
        # Apply SIREN-specific initialization
        self.net.apply(nl_weight_init)  # Hidden layer init
        self.net[0].apply(first_layer_init)  # First layer init
```

**Network architecture for Dubins3D:**
```
Input (t,x,y,θ) ∈ ℝ⁴
    ↓
Linear(4 → 256) + sin(30·)  [first_layer_sine_init]
    ↓
Linear(256 → 256) + sin(30·)  [sine_init]
    ↓
Linear(256 → 256) + sin(30·)  [sine_init]
    ↓
Linear(256 → 256) + sin(30·)  [sine_init]
    ↓
Linear(256 → 1)  [outermost_linear]
    ↓
Output: scalar V̂(t,x,y,θ)
```

### Initialization Details

#### First Layer Initialization

From lines 155-161:
```python
def first_layer_sine_init(m):
    """Map input [-1, 1] to reasonable sin input range"""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)  # = 4 for Dubins
            # w ∈ [-1/4, 1/4] × 256
            m.weight.uniform_(-1 / num_input, 1 / num_input)
```

**Effect:**
```
Input: t ∈ [0, 1], x,y ∈ [-1, 1], θ ∈ [-π·α, π·α]  (normalized)
    ↓
After Linear(4 → 256) with w ∈ [-1/4, 1/4]:
    z = w·[t, x, y, θ] ∈ [-1.4, 1.4]  (roughly)
    ↓
After sin(30·):
    sin(30·z) ∈ [sin(-42), sin(42)] ≈ [-0.99, 0.99]
    ↓
Good: Uses full sin range, derivative = 30·cos(.) ∈ [-30, 30]
```

#### Hidden Layer Initialization

From lines 150-154:
```python
def sine_init(m):
    """Maintain gradient magnitude through layers"""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)  # = 256
            # w ∈ [-√(6/256)/30, √(6/256)/30]
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / 30,  
                np.sqrt(6 / num_input) / 30
            )
```

**Magnitude calculation:**
```
√(6/256)/30 = √0.0234/30 = 0.1530/30 ≈ 0.0051

So w ∈ [-0.0051, 0.0051]  (very small weights!)

Effect:
After Linear(256 → 256) with small w:
    z = w·[sin(30·prev)] ≈ 0.0051 · sin(.)
    ↓
After sin(30·z) = sin(30 · 0.0051 · sin(.))
              = sin(0.153 · sin(.))
              ≈ 0.153 · sin(.)  (for small angles)
    ↓
Gradient: d/dz sin(30z) ≈ 30 at layer input, but weighted by small w
          Maintains constant gradient magnitude
```

---

## Step 2: Forward Pass (Evaluation Phase)

### File: `utils/modules.py` lines 127-136

```python
def forward(self, model_input, params=None):
    if params is None:
        params = OrderedDict(self.named_parameters())
    
    # Keep coords in computation graph for gradient computation
    coords_org = model_input['coords'].clone().detach().requires_grad_(True)
    coords = coords_org
    
    # Pass through SIREN layers
    output = self.net(coords)  # coords.shape = [batch, 4]
    
    return {
        'model_in': coords_org,      # Input (needed for gradient)
        'model_out': output           # Output scalar V̂
    }
```

**Key: `requires_grad_(True)` on input** 
- Makes input part of computation graph
- Allows automatic differentiation through network

### Example Forward Pass for One Sample

```
Input: (t=0.5, x=0.2, y=-0.1, θ=0.3) normalized coords

Layer 1:
  z₁ = Linear(0.5, 0.2, -0.1, 0.3)      [shape: [256]]
  z₁[i] ∈ [-1.4, 1.4]
  a₁ = sin(30 · z₁)                      [shape: [256]]
  a₁[i] ∈ [-1, 1] smooth

Layer 2:
  z₂ = Linear(a₁)                        [shape: [256]]
  z₂[i] ∈ [-0.2, 0.2]  (due to small init)
  a₂ = sin(30 · z₂)                      [shape: [256]]
  a₂[i] ∈ [-0.99, 0.99]  (but smooth!)

Layer 3:
  z₃ = Linear(a₂)
  a₃ = sin(30 · z₃)

Layer 4:
  z₄ = Linear(a₃)
  a₄ = sin(30 · z₄)

Output:
  V̂ = Linear(a₄)                         [shape: [1]]
  V̂ ∈ [-1, 1]  (normalized)
```

**Critical property:** Each hidden output $a_l$ is **smooth** because:
- $\sin$ is smooth
- All intermediate values bounded
- No ReLU discontinuities!

---

## Step 3: Gradient Computation (THE CRITICAL STEP)

### File: `dynamics/dynamics.py` lines 70-90

This is where SIREN's smoothness is exploited:

```python
def io_to_dv(self, input, output):
    """Compute spatial gradients ∇V from network output"""
    
    # Step 1: Compute Jacobian matrix ∂(output)/∂(input)
    #         This uses automatic differentiation through SIREN
    dodi = jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)
    # dodi.shape = [batch, 4]
    # dodi = [∂output/∂t, ∂output/∂x, ∂output/∂y, ∂output/∂θ]
    
    # Step 2: Scale from network units to physical units
    if self.deepreach_model == "exact":
        # Time derivative
        dvdt = (self.value_var / self.value_normto) * \
               (input[..., 0]*dodi[..., 0] + output)
        
        # Spatial gradient (corrected for scaling and change of variables)
        dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * \
               dodi[..., 1:] * input[..., 0].unsqueeze(-1)
        # dvds.shape = [batch, 3]
        # dvds = [∂V/∂x, ∂V/∂y, ∂V/∂θ]
    
    # Step 3: Concatenate and return
    return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)
    # Returns [∂V/∂t, ∂V/∂x, ∂V/∂y, ∂V/∂θ]
```

### How Automatic Differentiation Computes Smooth Gradients

The key function (from `utils/diff_operators.py` lines 7-24):

```python
def jacobian(y, x):
    """Compute dy/dx via automatic differentiation"""
    jac = torch.zeros(*y.shape, x.shape[-1]).to(y.device)
    
    for i in range(y.shape[-1]):  # For each output element
        y_flat = y[..., i].view(-1, 1)
        
        # Compute gradient: torch.autograd.grad traces through SIREN
        jac[..., i, :] = grad(y_flat, x, 
                             torch.ones_like(y_flat), 
                             create_graph=True)[0]
```

**What happens inside `grad()`:**

The chain rule traces through all SIREN layers:

```
∂V/∂t = (∂V/∂a₄) · (∂a₄/∂z₄) · (∂z₄/∂a₃) · ... · (∂a₁/∂t)

Where each term has the form:
  ∂aₗ/∂zₗ = cos(30·zₗ) · 30  ← Always bounded and smooth!
  ∂zₗ/∂aₗ₋₁ = Wₗ             ← Linear, smooth
  
Final result: Product of smooth, bounded terms = SMOOTH GRADIENT
```

**Compare with ReLU:**

```
∂V/∂t = (∂V/∂a₄) · (step function) · (∂z₄/∂a₃) · (step function) · ...

Where step function:
  ∂aₗ/∂zₗ = {0 if zₗ < 0, 1 if zₗ > 0}  ← DISCONTINUOUS!
  
Final result: Product with step functions = PIECEWISE CONSTANT
```

---

## Step 4: Hamiltonian Computation (Using ∇V)

### File: `dynamics.py` - Example from Dubins3D

From your code (~line 300 in dynamics.py):

```python
def hamiltonian(self, state, dvds):
    """Compute H(x, ∇V) = max_u min_d [-∇V · f(x,u,d)]"""
    
    # For Dubins3D:
    # state = [x, y, θ], dvds = [∂V/∂x, ∂V/∂y, ∂V/∂θ]
    # dynamics: ẋ = -v + v·cos(θ) + u·y
    #           ẏ = v·sin(θ) - u·x  
    #           θ̇ = -u
    
    # Compute control term: u · (∂V/∂x · y - ∂V/∂y · x - ∂V/∂θ)
    control_term = dvds[..., 0] * state[..., 1] - \
                   dvds[..., 1] * state[..., 0] - \
                   dvds[..., 2]
    
    # Maximum over control: max_u becomes |u_max| · |control_term|
    ham = self.omega_max * torch.abs(control_term)
    
    # Add constant drift terms
    ham = ham + (self.velocity * (torch.cos(state[..., 2]) - 1.0) * dvds[..., 0]) + \
               (self.velocity * torch.sin(state[..., 2]) * dvds[..., 1])
    
    return ham
```

**Key observation:** This computation uses **smooth dvds**!

If dvds were piecewise constant (ReLU):
- `torch.abs(control_term)` would have jumps
- Hamiltonian would be discontinuous
- Loss would be noisy and hard to minimize

With SIREN smooth dvds:
- `torch.abs(control_term)` is smooth
- Hamiltonian is smooth
- Loss landscape is clean

---

## Step 5: Loss Function (Enforcing HJ Constraint)

### File: `utils/losses.py` lines 4-18

```python
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
        
        # Compute Hamiltonian using SMOOTH gradients dvds
        ham = dynamics.hamiltonian(state, dvds)  # ← Uses smooth ∇V
        
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)
        
        # Main HJ constraint: ∂V/∂t + H(x, ∇V) = 0
        diff_constraint_hom = dvdt - ham
        
        if minWith == 'target':
            # For reach: also enforce V ≥ boundary_value
            diff_constraint_hom = torch.max(
                diff_constraint_hom, 
                value - boundary_value
            )
        
        # Boundary condition: V = boundary_value on boundary
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        
        return {
            'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
            'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()
        }
```

**Loss landscape with SIREN:**

```
For a batch of 60,000 training points:

term 1: Σ |∂V/∂t + H(x, ∇V)|
        = Σ |smooth_term + smooth_term|
        = smooth function of network weights

term 2: Σ |V - boundary|
        = smooth function of network weights

Total loss: smooth, continuous, differentiable!

Gradient descent: ∇_w L is stable and points in good directions
```

**Loss landscape with ReLU (what would happen):**

```
For the same batch:

term 1: Σ |∂V/∂t + H(x, ∇V)|
        = Σ |piecewise_const + mixed_terms|
        = noisy function with discontinuities at ReLU boundaries

Gradient descent: ∇_w L has sparse, noisy gradients
                 Training converges slowly or gets stuck
```

---

## Step 6: Training (Gradient Descent)

### File: `experiments.py` lines 115-160

```python
def train(self, device, batch_size, epochs, lr, ...):
    optim = torch.optim.Adam(lr=lr, params=self.model.parameters())
    
    for epoch in range(0, epochs):
        for step, (model_input, gt) in enumerate(train_dataloader):
            # Forward pass through SIREN
            model_results = self.model({'coords': model_input['model_coords']})
            
            # Convert outputs to physical units
            states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
            values = self.dataset.dynamics.io_to_value(...)
            
            # THE CRITICAL STEP: Compute smooth gradients via autodiff
            dvs = self.dataset.dynamics.io_to_dv(
                model_results['model_in'],  # Input (needs grad)
                model_results['model_out']   # Output from SIREN
            )
            # dvs = [∂V/∂t, ∂V/∂x, ∂V/∂y, ∂V/∂θ]  ← All smooth!
            
            boundary_values = gt['boundary_values']
            dirichlet_masks = gt['dirichlet_masks']
            
            # Compute loss using smooth gradients
            losses = loss_fn(states, values, 
                           dvs[..., 0],      # ∂V/∂t (smooth)
                           dvs[..., 1:],    # ∇_x V (smooth)
                           boundary_values, dirichlet_masks, 
                           model_results['model_out'])
            
            # Backpropagation through smooth loss landscape
            optim.zero_grad()
            train_loss = 0.
            for loss_name, loss in losses.items():
                train_loss += loss.mean()
            train_loss.backward()  # ← Traces back through SIREN
            
            optim.step()  # ← Weight update based on smooth gradients
```

**Why SIREN makes this work:**

1. **Smooth forward pass** → smooth gradients via autodiff
2. **Smooth loss** → clean optimization trajectory
3. **Clean gradients** → efficient weight updates
4. **Result** → Value function converges to true HJ solution

---

## Step 7: Optimal Control Extraction

### File: `dynamics.py` - Example from Dubins3D

```python
def optimal_control(self, state, dvds):
    """Given ∇V, directly compute optimal control u*"""
    
    # For Dubins3D: u* = sign(∂V/∂x · y - ∂V/∂y · x - ∂V/∂θ)
    det = dvds[..., 0]*state[..., 1] - \
          dvds[..., 1]*state[..., 0] - \
          dvds[..., 2]
    
    # Optimal control: u* ∈ {-ω_max, +ω_max} sign
    return (self.omega_max * torch.sign(det))[..., None]
```

**The Safety Property:**

With SIREN smooth dvds:
```
dvds(x) is Lipschitz continuous
    ↓
det(x) = function of dvds(x) is Lipschitz
    ↓
sign(det) is "almost smooth" (only jumps at det=0, isolated points)
    ↓
u*(x) = ω_max · sign(det) is safe
```

With ReLU piecewise constant dvds:
```
dvds(x) jumps at ReLU boundaries (lots of them!)
    ↓
det(x) has many discontinuities
    ↓
sign(det) jumps frequently
    ↓
u*(x) has many high-frequency oscillations ← UNSAFE!
```

---

## Complete Inference Pipeline

Here's how to use the trained SIREN for safe control synthesis:

```python
# 1. Load trained model
model = SingleBVPNet(out_features=1, type='sine', in_features=4,
                    hidden_features=256, num_hidden_layers=3)
model.load_state_dict(torch.load('model_final.pth'))
model.eval()

# 2. Current state
state = torch.tensor([[x, y, theta]])  # Physical units

# 3. Normalize coordinates
normalized_input = dynamics.coord_to_input(state)

# 4. Forward pass through SIREN
model_results = model({'coords': normalized_input})

# 5. Compute smooth spatial gradients
dvs = dynamics.io_to_dv(model_results['model_in'], 
                        model_results['model_out'])
grad_V = dvs[..., 1:]  # ∇_state V  ← SMOOTH gradient!

# 6. Compute optimal control
u_optimal = dynamics.optimal_control(state, grad_V)

# 7. Apply control: u* is SMOOTH and SAFE
print(f"Optimal control: {u_optimal}")  # ← Implementable on real system!
```

---

## Comparison: What Would Go Wrong Without SIREN

### Using ReLU Instead

```python
# Using ReLU network (wrong choice for this problem)
model = SingleBVPNet(..., type='relu')  # ← Bad idea!

# Everything up to gradient computation...
# But then:

dvs = dynamics.io_to_dv(model_results['model_in'], 
                        model_results['model_out'])
# dvs computes gradients through ReLU
# ∂V/∂x contains jumps at ReLU boundaries!

grad_V = dvs[..., 1:]  # ∇_state V  ← PIECEWISE CONSTANT!

u_optimal = dynamics.optimal_control(state, grad_V)
# u_optimal oscillates at different regions
# Real system would experience jerky, unsafe control!
```

---

## Summary: The SIREN-Safe Control Pipeline

```
Input (t,x) 
    ↓
[SIREN Network with sin(30·) activations]
    ↓
Output Ṽ(t,x)  ← Approximates true V
    ↓
[Automatic Differentiation (smooth!)]
    ↓
∇Ṽ = [∂V/∂t, ∂V/∂x]  ← Smooth, continuous gradients
    ↓
[Hamiltonian Computation]
    ↓
H(x, ∇Ṽ)  ← Smooth function
    ↓
[Control Synthesis from Gradient]
    ↓
u* = f(∇Ṽ)  ← Lipschitz continuous, safe control!
    ↓
Real System Execution (smooth, robust)
```

This pipeline works because SIREN ensures smoothness at every step.
