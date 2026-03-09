# Baselines

Ground-truth Hamilton-Jacobi reachability value functions using `optimized_dp` for benchmarking against DeepReach.

## Setup

We import `optimized_dp` as an external dependency rather than absorbing since it includes HeteroCL which is too large to vendor.

1. Clone optimized_dp:
```bash
git clone https://github.com/SFU-MARS/optimized_dp.git ~/optimized_dp
```

2. Create the conda env:
```bash
cd ~/optimized_dp
conda env create -f environment.yml
conda activate odp
pip install -e .
```

## Running the solvers

Make sure you're in the `odp` conda env and in the `deepreach_CMPT419` root:

```bash
conda activate odp
cd ~/deepreach_CMPT419

python baselines/optimized_dp/air3d_solve.py
python baselines/optimized_dp/dubins3d_solve.py
```

## Outputs

- `baselines/grids/air3d_grid.npy` — Air3D value function (101x101x101)
- `baselines/grids/air3d_metadata.json` — params, timing, memory
- `baselines/grids/dubins3d_grid.npy` — Dubins3D value function (101x101x101)
- `baselines/grids/dubins3d_metadata.json` — params, timing, memory

## Comparing against DeepReach

`compare_values.py` supports two modes:

**Mode A** — compare .npy vs .npy (no GPU needed):
```bash
conda activate odp
python baselines/compare_values.py \
  --baseline_grid baselines/grids/air3d_grid.npy \
  --deepreach_grid path/to/deepreach_air3d_values.npy
```

**Mode B** — compare .npy vs .pth model (needs PyTorch):
```bash
conda activate deepreach
python baselines/compare_values.py \
  --baseline_grid baselines/grids/air3d_grid.npy \
  --deepreach_model path/to/model_final.pth \
  --dynamics air3d
```

Outputs to `baselines/plots/`: `slice_comparison.png` and `brt_overlap.png`.

## Parameters

All shared parameters are in `baselines/config.py`. These match the DeepReach dynamics in `dynamics/dynamics.py`.
