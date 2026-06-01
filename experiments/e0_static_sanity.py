"""
Static TSP overfit check.
Goal: verify that Neuroevolution can learn the optimal tour on a tiny fixed instance.
Not used in the paper — purely a framework sanity check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import wandb
import pandas as pd

from environment.tdtsp import TDTSPEnv
from models.policy_net import PolicyNet
from solvers.ga import run_ga
from solvers.neuroevolution import run_neuroevolution
from utils.metrics import random_baseline, brute_force_optimal

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N = 6
SEED = 42
GA_POP = 100
GA_GENS = 300
NE_POP = 50
NE_GENS = 500
NE_SIGMA = 0.1
HIDDEN = 128

env = TDTSPEnv(n_cities=N, seed=SEED, perturbation_fraction=0.0)

print(f"=== Static TSP Overfit Check | n={N}, seed={SEED} ===\n")

rand_mean, rand_std = random_baseline(env, n_samples=500)
print(f"Random baseline   : {rand_mean:.4f} ± {rand_std:.4f}")

opt_tour, opt_cost = brute_force_optimal(env)
print(f"Optimal (brute)   : {opt_cost:.4f}  tour={opt_tour}")

# --- GA on routes ------------------------------------------------------------
print("\nRunning GA (route-level)...")
ga_result = run_ga(env, pop_size=GA_POP, mutation_rate=0.1, n_generations=GA_GENS, seed=SEED)
print(f"GA eval cost      : {ga_result['eval_cost']:.4f}  ({ga_result['runtime']:.1f}s)")
gap_ga = (ga_result["eval_cost"] - opt_cost) / opt_cost * 100
print(f"GA gap to optimal : {gap_ga:.2f}%")
pd.DataFrame(ga_result["history"]).to_csv(RESULTS / "e0_ga.csv", index=False)

# --- Neuroevolution on policy weights ----------------------------------------
print("\nRunning Neuroevolution (weight-level)...")
policy = PolicyNet(n_cities=N, hidden_dim=HIDDEN)

wandb.init(
    project="aci-dtsp",
    name=f"e0-neuroevo-n{N}-seed{SEED}",
    config=dict(n_cities=N, seed=SEED, pop_size=NE_POP, n_generations=NE_GENS,
                sigma=NE_SIGMA, hidden_dim=HIDDEN, perturbation_fraction=0.0),
    mode="online",
)

ne_result = run_neuroevolution(
    env, policy,
    pop_size=NE_POP,
    sigma=NE_SIGMA,
    n_generations=NE_GENS,
    elite_frac=0.2,
    tournament_k=3,
    crossover=True,
    seed=SEED,
    val_every=10,
)
wandb.finish()

print(f"Neuroevo cost     : {ne_result['final_cost']:.4f}  ({ne_result['runtime']:.1f}s)")
gap_ne = (ne_result["final_cost"] - opt_cost) / opt_cost * 100
print(f"Neuroevo gap      : {gap_ne:.2f}%")
pd.DataFrame(ne_result["history"]).to_csv(RESULTS / "e0_neuroevo.csv", index=False)

# --- Summary -----------------------------------------------------------------
print("\n=== Summary ===")
print(f"{'Method':<22} {'Cost':>8} {'Gap to Opt':>12}")
print(f"{'Random':<22} {rand_mean:>8.4f} {'—':>12}")
print(f"{'Brute-force opt':<22} {opt_cost:>8.4f} {'0.00%':>12}")
print(f"{'GA (route-level)':<22} {ga_result['eval_cost']:>8.4f} {gap_ga:>11.2f}%")
print(f"{'Neuroevolution':<22} {ne_result['final_cost']:>8.4f} {gap_ne:>11.2f}%")
print("\nResults saved to results/e0_*.csv")
