# Neuroevolution vs. Genetic Algorithms for a Dynamic TSP

Software prototype for the FHNW MSc Medical Informatics ACI paper *"A Comparison of
Neuroevolutionary Reinforcement Learning and Genetic Algorithms for a Dynamic
Travelling Salesman Problem"*.

The code compares two adaptive, gradient-free optimization methods on a
**Time-Dependent Travelling Salesman Problem (TDTSP)** where edge costs change over
time (rush-hour cost spikes on random edges):

1. **Genetic Algorithm (GA)** — directly evolves specific tours for an instance.
2. **Deep Neuroevolution** — a GA optimizes the weights of an attention policy network
   that produces tours, learning a generalized routing policy (Such et al. 2018).
   A **REINFORCE** policy-gradient baseline is also included.

All experiments are reproducible from `params.yaml`, log to Weights & Biases, and write
CSV data to `results/` and matplotlib figures to `figures/`.

## Project Structure

```
aci_paper/
├── params.yaml              # Single source of truth for all hyperparameters
├── run_all.py               # Runs E1–E4 in sequence under one shared run ID
├── pyproject.toml           # Poetry project + dependencies
│
├── src/
│   ├── environment/
│   │   └── tdtsp.py         # TDTSPEnv — Gymnasium env, time-varying cost matrix
│   ├── models/
│   │   └── policy_net.py    # Attention-based routing policy network
│   ├── solvers/
│   │   ├── ga.py            # Standard GA over tours (direct route evolution)
│   │   ├── neuroevolution.py# GA over policy-net weights (Deep Neuroevolution)
│   │   └── reinforce.py     # REINFORCE policy-gradient baseline
│   └── utils/
│       ├── config.py        # Loads params.yaml, manages shared run ID
│       ├── metrics.py       # Reference tour (NN + 2-opt), gap-to-reference
│       └── visualization.py # Tour GIFs / convergence plots
│
├── experiments/
│   ├── e0_static_sanity.py  # Static-TSP overfit check (framework sanity, not in paper)
│   ├── e1_ga_sensitivity.py # E1: 3×3 grid over pop_size × mutation_rate (heatmaps)
│   ├── e2_neuroevo_stability.py # E2: 3 seeded runs → learning curve with variance
│   ├── e3_scaling.py        # E3: best GA vs. best policy across instance sizes
│   └── e4_head_to_head.py   # E4: final GA vs. RL comparison on canonical instance
│
├── results/                 # CSV output, written per run ID
└── figures/                 # PNG/PDF/GIF output, written per run ID
```

## Setup

Requires **Python ≥ 3.12** and **Poetry**. The Torch build is pinned to CUDA 12.6
wheels for Windows; on other platforms Poetry resolves the CPU/standard build.

```bash
# 1. Install dependencies into a virtual environment
poetry install

# 2. (Optional) log in to Weights & Biases, or disable it (see below)
poetry run wandb login
```

If you do not want experiment tracking, set `wandb.mode: "disabled"` in `params.yaml`.

## Running the Experiments

Run everything end-to-end (results and figures land in `results/<RUN_ID>/` and
`figures/<RUN_ID>/`, where `RUN_ID` is a shared timestamp):

```bash
poetry run python run_all.py
```

Or run a single experiment:

```bash
poetry run python experiments/e1_ga_sensitivity.py
poetry run python experiments/e2_neuroevo_stability.py
poetry run python experiments/e3_scaling.py
poetry run python experiments/e4_head_to_head.py
```

| Experiment | Question | Output |
|------------|----------|--------|
| **E1** | GA sensitivity to `pop_size` × `mutation_rate` | Cost/gap/runtime heatmaps |
| **E2** | RL/neuroevolution training stability across seeds | Learning curve with variance band |
| **E3** | How each method scales with instance size | Grouped bar + log-scale time charts |
| **E4** | Best GA vs. best RL head-to-head | Cost, % gap, inference time summary |

## Configuration

All hyperparameters live in `params.yaml` — instance size, perturbation level, GA grid,
neuroevolution settings, per-experiment overrides, and the W&B project/mode. Edit that
file to define new test cases; the experiment scripts read from it directly.
