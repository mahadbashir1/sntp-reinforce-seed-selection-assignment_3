# SE4095 Assignment 03 - Seed Selection with REINFORCE

Mahad Bashir (23F3011)

An RL agent selects 5 seed users in the SocioPatterns workplace contact graph (92 participants, 755 edges).
Spread is simulated with the Independent Cascade model (p = 0.10). An MLP policy and a Transformer policy are
trained with REINFORCE and compared with Random and Degree selection on the same 100 diffusion seeds.

## Files

| File | Content |
|---|---|
| `environment.py` | graph loading, Independent Cascade simulator, RL environment (`reset`, `step`) |
| `models.py` | MLP policy, Transformer policy, masked action distribution, greedy action |
| `reinforce.py` | REINFORCE training loop with running baseline |
| `main.py` | runs everything: network analysis, checks, training, evaluation, CSVs and figures |
| `data/` | supplied dataset package (unchanged) |
| `results/` | all outputs produced by `main.py` |
| `report.pdf` | report |

## Installation

Python 3.11 was used.

```
pip install -r requirements.txt
```

`requirements.txt` has the exact versions used: numpy 1.26.4, pandas 2.2.3, networkx 3.5, matplotlib 3.10.0,
torch 2.11.0 (CPU build).

## Running

From this folder:

```
python main.py --output results
```

This creates:

- `results/graph_edges.csv`, `training.csv` (6000 rows), `evaluation.csv` (800 rows), `summary.csv` (8 rows)
- `results/checks.json` - pass/fail of the required checks
- `results/analysis.json` - network statistics, top-10 degree table, architecture-level mean/std,
  selected users and department coverage
- `results/checkpoints/` - 6 final checkpoints (parameters + architecture settings)
- `results/figures/` - network, learning curves, final spread comparison, Transformer seed 7 selection

## Hardware and runtime

- Laptop: Intel Core i7-10750H (2.6 GHz, 6 cores), 16 GB RAM, Windows 11
- Everything runs on CPU (CPU build of PyTorch). No GPU is used, so device synchronization is not needed
  before timing.
- Full run takes about 1 minute (MLP run ~5 s, Transformer run ~10-15 s, evaluation a few seconds).

## Reproducibility

- Seeds follow the assignment: Python, NumPy and PyTorch seeds are set to the training seed at the start of each
  run, training diffusion seeds are `1000000 + 10000*s + e`, evaluation seeds are 900000-900099, and Random uses
  `default_rng(diffusion_seed + 10000000)`.
- Running `main.py` twice gave identical `training.csv`, `evaluation.csv` and `summary.csv` except the timing
  columns (`selection_seconds`, `training_seconds`), which depend on the machine load.

## Notes / incomplete requirements

- All required parts are implemented. No requirement is knowingly left incomplete.
- On my Windows machine, importing pandas before torch gave a DLL error (`WinError 1114`), so `main.py` imports
  torch first.
- Selection time = time the method takes to choose its 5 users (policy forward passes / sorting / sampling).
  Environment steps and the diffusion simulation are not included in it.
