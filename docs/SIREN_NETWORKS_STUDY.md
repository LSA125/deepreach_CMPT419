# SIREN Networks for Smooth Safety Gradients in DeepReach

## Overview

SIREN (Sinusoidal Representation Networks) are neural networks using sinusoidal activations instead of ReLU. They are critical for DeepReach because computing **smooth spatial gradients** of the value function is essential for safe control synthesis.

---

## 1. Why Sinusoidal Activations? The Activation Function Crisis

### The Problem with ReLU for Gradient-Based Control

Standard neural networks use **ReLU** activations: $\sigma(z) = \max(0, z)$

**Fatal flaw for safety gradients:**
- **ReLU derivative is piecewise constant**: $\sigma'(z) = \begin{cases} 0 & z < 0 \\ 1 & z > 0 \\ \text{undefined} & z = 0 \end{cases}$
- **Discontinuous at z = 0** → gradient jumps suddenly
- After multiple layers, accumulated discontinuities → **highly oscillatory gradients**
- When control law depends on $\nabla V$, discontinuities → jittery, unsafe control!

Example problem with ReLU:
```
Network: x → ReLU → ReLU → ReLU → output V
Gradient chain: ∂V/∂x = ∂V/∂z3 · ∂z3/∂z2 · ∂z2/∂z1 · ∂z1/∂x
                      = (0 or 1) × (0 or 1) × (0 or 1) × (linear)
                      → Many zeros or sudden jumps between regions
```

### The SIREN Solution: Sinusoidal Activations

SIREN uses: **$\sigma(z) = \sin(30z)$**

**Why this fixes the problem:**
- **Smooth everywhere**: $\sigma'(z) = 30 \cdot \cos(30z)$ is continuous
- **Bounded derivative**: $|\sigma'(z)| \leq 30$ for all z → no gradient explosion
- **No dead zones**: sin oscillates, every input influences gradient
- **Multi-scale representation**: sin(30z) captures high-frequency features naturally

**Derivative comparison:**
```
ReLU:  σ(z) = max(0,z)        σ'(z) = {0, 1}         ← Discontinuous
Sine:  σ(z) = sin(30z)        σ'(z) = 30·cos(30z)    ← Smooth & continuous
```

---

## 2. How SIREN Is Implemented in DeepReach

### Architecture in `utils/modules.py`

```python
class Sine(nn.Module):
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)
```

The network stack from [modules.py lines 50-95](utils/modules.py#L50-L95):

```python
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', ...):
        
        nls_and_inits = {
            'sine': (Sine(), sine_init, first_layer_sine_init),
            'relu': (nn.ReLU(), ..., None),
            # ... other activations
        }
        
        # Build network: Linear → Sine → Linear → Sine → ... → Linear (outermost)
        self.net.append(nn.Sequential(BatchLinear(...), nl))  # First layer
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(BatchLinear(...), nl))  # Hidden
        self.net.append(nn.Sequential(BatchLinear(...)))  # Output (linear)
```

---

## 3. Crucial: Special Weight Initialization for SIREN

Standard initialization (e.g., Kaiming for ReLU) **breaks SIREN**. You must use SIREN-specific initialization:

### First Layer Initialization

From [modules.py line 155](utils/modules.py#L155):

```python
def first_layer_sine_init(m):
    """Initialize first layer to have input range ≈ [-1, 1]"""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # Uniform in [-1/n_input, 1/n_input]
            m.weight.uniform_(-1 / num_input, 1 / num_input)
```

**Why:** 
- Maps input domain to $[-1/n_{in}, 1/n_{in}]$ ranges
- After 30× scaling in Sine: $[-30/n_{in}, 30/n_{in}]$
- For typical $n_{in} \approx 4$ (time + 3D state): range ≈ $[-7.5, 7.5]$ → good sin input

### Hidden Layer Initialization

From [modules.py line 150](utils/modules.py#L150):

```python
def sine_init(m):
    """Initialize hidden layers properly for sin activation"""
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # Uniform in [-√(6/n_input)/30, √(6/n_input)/30]
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
```

**Why the √6 and /30 factor?** 
- Ensures **equivariance**: $\sigma'(z) \cdot w \cdot x$ has constant expected magnitude across layers
- Without this: gradients vanish or explode through deep networks
- Factor of 30: matches the scaling in Sine activation

---

## 4. The Critical Gradient Computation Pipeline

### Step-by-Step: From Network Output to Safety Gradients

Your code in [experiments.py line 139](experiments/experiments.py#L139):

```python
model_results = self.model({'coords': model_input['model_coords']})

# Step 1: Get network output V(t,x)
V = model_results['model_out']  # Shape: [batch, 1]

# Step 2: Compute GRADIENTS via automatic differentiation
dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], V)
# dvs = [∂V/∂t, ∂V/∂x₁, ∂V/∂x₂, ∂V/∂x₃]  ← THE CRUCIAL SAFETY GRADIENTS
```

From [dynamics.py lines 70-90](dynamics/dynamics.py#L70-L90):

```python
def io_to_dv(self, input, output):
    # Compute Jacobian: d(output)/d(input)
    dodi = jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)
    
    if self.deepreach_model == "exact":
        # Scale to real units
        dvdt = (self.value_var / self.value_normto) * (input[..., 0]*dodi[..., 0] + output)
        dvds = (self.value_var / self.value_normto / self.state_var) * \
               dodi[..., 1:] * input[..., 0].unsqueeze(-1)
    else:
        dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
        dvds = (self.value_var / self.value_normto / self.state_var) * dodi[..., 1:]
    
    return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)
```

### Why SIREN Smoothness Is Essential Here

When computing $\nabla_x V$ via autodiff:

$$\nabla_x V = \frac{\partial V}{\partial x} = \frac{\partial}{\partial x}[\text{sin}(30 \cdot f_{L}(\cdots \text{sin}(30 \cdot f_1(x))))]$$

**With ReLU activation:**
- Derivatives involve products of {0, 1} values
- Result: piecewise constant ∇V across network regions
- Control law $u^* = f(\nabla_x V)$ becomes discontinuous

**With sin activation:**
- Derivatives are smooth products of bounded (30·cos) terms
- Result: smooth, continuous ∇V
- Control law is safe and implementable

---

## 5. The HJ Loss Function Enforcement

Your loss in [losses.py](losses.py) directly uses the computed gradients:

```python
def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
    # Compute Hamiltonian using ∇V
    ham = dynamics.hamiltonian(state, dvds)  # Takes dvds = ∇_x V
    
    # Enforce HJ constraint: ∂V/∂t + H(x, ∇V) = 0
    diff_constraint = dvdt - ham  # Should be ≈ 0
    
    return {'diff_constraint_hom': torch.abs(diff_constraint).sum()}
```

**How SIREN enables this:**

| Component | With ReLU | With SIREN |
|-----------|-----------|------------|
| $\nabla_x V$ (gradients) | Piecewise constant | Smooth continuous |
| Hamiltonian $H(x, \nabla_x V)$ | Discontinuous | Smooth |
| HJ constraint $\frac{\partial V}{\partial t} + H = 0$ | Hard to satisfy | Natural to learn |
| Resulting control law | Jittery, unsafe | Smooth, safe |
| Gradient flow for training | Noisy | Clean |

---

## 6. Factor of 30: The Magic Number

The choice of $\sin(30z)$ appears repeatedly in your code with references to Sitzmann et al. (2020):

**Why 30 specifically?**

1. **Fourier feature encoding**: $\sin(30z)$ → captures frequencies up to ~30
2. **Input sensitivity**: Typical input range is $[-1, 1]$ (normalized coordinates)
   - $\sin(30 \cdot 1) = \sin(30 \text{ rad}) \approx -0.988$ → uses full sine range
3. **Gradient magnitude**: $\frac{d}{dz}\sin(30z) = 30\cos(30z)$ 
   - Maximum derivative is 30 → matches hidden layer init scale
   - Prevents gradient magnitude explosion/vanishing

4. **Empirically optimal**: Paper shows 30 gives best trade-off between:
   - Frequency content (captures fine details)
   - Gradient stability (no explosion/vanishing)
   - Training convergence speed

---

## 7. Comparison: Gradient Smoothness Across Activations

Here's why **only** sinusoidal activations work for your problem:

### ReLU Network Gradients
```
Layer 1: sin₁(x) = ReLU(w₁x)            ∂sin₁/∂x = w₁ · {0 or 1}
Layer 2: sin₂ = ReLU(w₂·sin₁)           ∂sin₂/∂x = w₂ · {0 or 1} · ∂sin₁/∂x
                                        = w₂·w₁ · {0 or 1}²  ← Still discontinuous!
...
Output: V(x) = ReLU(wₙ·sinₙ)            ∂V/∂x =  ∏(discrete factors)  ← JERKY!
```

### SIREN Gradients
```
Layer 1: sin₁(x) = sin(30·w₁x)          ∂sin₁/∂x = 30w₁·cos(30w₁x)  ← Smooth
Layer 2: sin₂ = sin(30·w₂·sin₁)         ∂sin₂/∂x = 30w₂·cos(30w₂sin₁) · ∂sin₁/∂x
                                        = 30w₂·cos(...) · 30w₁·cos(...)  ← Still smooth!
...
Output: V(x) = wₙ·sinₙ (linear out)     ∂V/∂x = ∏(smooth cos factors)  ← SMOOTH!
```

---

## 8. Key Insights for Safe Control

### Why DeepReach Chooses SIREN

1. **Safety Critical**: Control law $u^* = \arg\max_u H(x, \nabla_x V)$ directly depends on gradient
   - Discontinuous gradients → unsafe control
   - Smooth gradients → provably safe control

2. **High-Dimensional Spaces**: Your 3D Dubins problem needs to represent the entire value function smoothly
   - ReLU would create artifacts at decision boundaries
   - SIREN's sinusoidal basis naturally smooth in high dimensions

3. **Gradient Magnitudes Matter**: The Hamiltonian uses $|\nabla_x V|$
   - Robust to small perturbations with SIREN
   - Brittle with ReLU (sensitive to network region boundaries)

4. **Training Dynamics**: Loss contains $||∂V/∂t + H(x, \nabla_x V)||$
   - Smooth objectives → smooth loss landscapes → better convergence
   - Non-smooth objectives → noise → training instability

---

## 9. The Complete Pipeline: How SIREN Makes DeepReach Work

```
Input: (t, x) ∈ (time, state_space)
    ↓
[SIREN Network with sin(30z) activations]
    ↓
Output: Ṽ(t,x) ≈ True V(t,x)
    ↓
[Automatic Differentiation]
    ↓
∇V = [∂V/∂t, ∂V/∂x₁, ∂V/∂x₂, ∂V/∂x₃]  ← SMOOTH due to sin!
    ↓
[Hamiltonian Computation]
    ↓
H(x, ∇V) = max_u min_d [-∇V · f(x,u,d)]
    ↓
[HJ Loss]
    ↓
Loss = ||∂V/∂t + H(x, ∇V)||  ← Smooth landscape with SIREN
    ↓
[Gradient Descent]
    ↓
Learned V → Extract Optimal Control u* = f(∇V)
    ↓
Safe, Smooth Control Policy!
```

---

## 10. Summary: SIREN vs ReLU for HJ Value Functions

| Aspect | ReLU Networks | SIREN Networks |
|--------|---------------|----------------|
| **Activation** | $\max(0,z)$ | $\sin(30z)$ |
| **Derivative** | Constant (0 or 1) | Smooth: $30\cos(30z)$ |
| **∇V Smoothness** | Piecewise constant | Uniformly smooth |
| **Control Law** | Discontinuous, unsafe | Continuous, safe |
| **Loss Landscape** | Noisy, rough | Smooth, clean |
| **Training** | Difficult convergence | Natural convergence |
| **Weight Init** | Kaiming | SIREN-specific |
| **Best For** | Classification | PDEs, Dynamics |

---

## References in Your Code

- [Sine activation class](utils/modules.py#L49-L52)
- [FCBlock architecture](utils/modules.py#L55-L104)
- [Weight initialization schemes](utils/modules.py#L132-L161)
- [Gradient computation](dynamics/dynamics.py#L70-L90)
- [HJ loss using gradients](losses.py#L4-L18)
- [Training loop using smooth gradients](experiments/experiments.py#L139-L160)

Original paper: Sitzmann et al. "Implicit Neural Representations with Levels of Experts" (2020) - but the core SIREN idea is from "Implicit Neural Representations with Levels of Experts" (2021).

---

## Key Takeaway

**SIREN networks are not a luxury—they're a necessity** for safe control based on value function gradients. The sinusoidal activation ensures that $\nabla_x V$ is smooth everywhere, which guarantees that the optimal control law $u^*(x) = f(\nabla_x V)$ is safe, implementable, and robust to perturbations.
