import torch
from dynamics.dynamics import NarrowPassage

dyn = NarrowPassage(avoid_fn_weight=0.5, avoid_only=False)
c = torch.randn(1, 11, requires_grad=True)
out = dyn.coord_to_input(c)
out.sum().backward()
print(c.grad)