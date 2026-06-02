"""
E4 — Head-to-Head
Best GA vs Neuroevolution on the canonical DTSP instance (params.yaml).
Reports absolute costs, percentage gap, optimization time, and inference time.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb

from environment.tdtsp import TDTSPEnv
from models.policy_net import AttentionPolicyNet
from solvers.ga import run_ga
from solvers.neuroevolution import run_neuroevolution, evaluate_greedy
from utils.metrics import reference_tour, gap_to_reference
from utils.config import load_params, params_path, run_id

P = load_params()
RUN_ID = run_id()

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / RUN_ID
FIGURES = ROOT / "figures" / RUN_ID
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

N            = P["e4"]["n_cities"]
SEED         = P["env"]["seed"]
HIDDEN       = P["neuroevolution"]["hidden_dim"]
NE_POP       = P["neuroevolution"]["pop_size"]
NE_SIGMA     = P["neuroevolution"]["sigma"]
NE_GENS      = P["e4"]["ne_n_generations"]
ELITE_FRAC   = P["neuroevolution"]["elite_frac"]
TOURNAMENT_K = P["neuroevolution"]["tournament_k"]
CROSSOVER    = P["neuroevolution"]["crossover"]
NHEADS       = P["neuroevolution"]["n_heads"]
NLAYERS      = P["neuroevolution"]["n_layers"]
TRAIN_K      = P["neuroevolution"]["train_instances"]
GA_GENS      = P["e4"]["ga_n_generations"]

# ── Load best configs ─────────────────────────────────────────────────────────
e1_csv = ROOT / "results" / RUN_ID / "e1_ga_sensitivity.csv"
if not e1_csv.exists():
    candidates = sorted((ROOT / "results").glob("*/e1_ga_sensitivity.csv"), reverse=True)
    e1_csv = candidates[0] if candidates else None

if e1_csv and e1_csv.exists():
    e1_df = pd.read_csv(e1_csv)
    best_e1 = e1_df.loc[e1_df["eval_cost"].idxmin()]
    GA_POP = int(best_e1["pop_size"])
    GA_MUT = float(best_e1["mutation_rate"])
    print(f"Best GA config from E1: pop={GA_POP}, mut={GA_MUT:.2f}")
else:
    print("WARNING: E1 results not found — using defaults from params.yaml")
    GA_POP = P["ga"]["pop_sizes"][1]
    GA_MUT = P["ga"]["mutation_rates"][1]

e2_csv = ROOT / "results" / RUN_ID / "e2_neuroevo_stability.csv"
if not e2_csv.exists():
    candidates = sorted((ROOT / "results").glob("*/e2_neuroevo_stability.csv"), reverse=True)
    e2_csv = candidates[0] if candidates else None

if e2_csv and e2_csv.exists():
    e2_df = pd.read_csv(e2_csv)
    best_seed = int(e2_df.loc[e2_df["final_cost"].idxmin(), "seed"])
    print(f"Best NE seed from E2: {best_seed}")
else:
    print("WARNING: E2 results not found — using seed from params.yaml")
    best_seed = P["neuroevolution"]["seeds"][0]

env = TDTSPEnv(n_cities=N, seed=SEED)

print(f"\n=== E4: Head-to-Head | n={N}, seed={SEED} ===\n")

print("Computing reference tour (NN + 2-opt)...")
ref_tour, ref_cost, ref_method = reference_tour(env)
print(f"Reference ({ref_method}): {ref_cost:.4f}\n")

# ── GA ────────────────────────────────────────────────────────────────────────
print(f"Running GA (pop={GA_POP}, mut={GA_MUT:.2f}, gens={GA_GENS})...")
ga_res = run_ga(
    env, pop_size=GA_POP, mutation_rate=GA_MUT, n_generations=GA_GENS, seed=SEED,
    gif_path=FIGURES / "e4_ga_tour.gif", wandb_log=False,
)
ga_cost = ga_res["eval_cost"]
ga_opt_time = ga_res["runtime"]

t0 = time.perf_counter()
ga_inf_cost = env.evaluate_tour(ga_res["best_tour"], start_t=0)
ga_inf_time = time.perf_counter() - t0

print(f"  cost={ga_cost:.4f}   opt_time={ga_opt_time:.1f}s   inf_time={ga_inf_time*1e3:.3f}ms")

# ── Neuroevolution ────────────────────────────────────────────────────────────
print(f"Running Neuroevolution (pop={NE_POP}, σ={NE_SIGMA}, gens={NE_GENS}, seed={best_seed})...")
torch.manual_seed(best_seed)
np.random.seed(best_seed)
policy = AttentionPolicyNet(n_cities=N, hidden_dim=HIDDEN,
                            n_heads=NHEADS, n_layers=NLAYERS)
train_envs = [TDTSPEnv(n_cities=N, seed=1000 + i) for i in range(TRAIN_K)]

wandb.init(
    project=P["wandb"]["project"],
    name=f"e4-head-to-head-n{N}-{RUN_ID}",
    config=dict(run_id=RUN_ID, n_cities=N, seed=SEED, ga_pop=GA_POP, ga_mut=GA_MUT,
                ga_gens=GA_GENS, ne_pop=NE_POP, ne_sigma=NE_SIGMA, ne_gens=NE_GENS,
                ne_seed=best_seed, elite_frac=ELITE_FRAC, tournament_k=TOURNAMENT_K,
                crossover=CROSSOVER),
    mode=P["wandb"]["mode"],
)
wandb.save(str(params_path()), policy="now")

ne_res = run_neuroevolution(
    env, policy, pop_size=NE_POP, sigma=NE_SIGMA, n_generations=NE_GENS,
    elite_frac=ELITE_FRAC, tournament_k=TOURNAMENT_K, crossover=CROSSOVER,
    seed=best_seed, val_every=P["neuroevolution"]["val_every"], wandb_log=False,
    gif_path=FIGURES / "e4_ne_tour.gif", train_envs=train_envs,
)
ne_cost = ne_res["final_cost"]
ne_opt_time = ne_res["runtime"]

t0 = time.perf_counter()
ne_inf_cost = evaluate_greedy(env, policy)
ne_inf_time = time.perf_counter() - t0

print(f"  cost={ne_cost:.4f}   opt_time={ne_opt_time:.1f}s   inf_time={ne_inf_time*1e3:.3f}ms")

# ── Results ───────────────────────────────────────────────────────────────────
gap_ne_vs_ga  = gap_to_reference(ne_cost,  ga_cost)
gap_ga_vs_ref = gap_to_reference(ga_cost,  ref_cost)
gap_ne_vs_ref = gap_to_reference(ne_cost,  ref_cost)

results = pd.DataFrame([
    {"method": "Reference (NN+2opt)", "cost": ref_cost, "gap_vs_ref_pct": 0.0,
     "opt_time_s": None, "inf_time_ms": None},
    {"method": "GA",             "cost": ga_cost, "gap_vs_ref_pct": gap_ga_vs_ref,
     "opt_time_s": ga_opt_time, "inf_time_ms": ga_inf_time * 1e3},
    {"method": "Neuroevolution", "cost": ne_cost, "gap_vs_ref_pct": gap_ne_vs_ref,
     "opt_time_s": ne_opt_time, "inf_time_ms": ne_inf_time * 1e3},
])
results.to_csv(RESULTS / "e4_head_to_head.csv", index=False)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
methods_3 = ["NN+2-opt\n(reference)", "GA", "Neuroevolution"]
costs_3   = [ref_cost, ga_cost, ne_cost]
colors_3  = ["#9E9E9E", "#2196F3", "#FF5722"]

bars = axes[0].bar(methods_3, costs_3, color=colors_3)
axes[0].set_ylabel("Tour cost")
axes[0].set_title(f"Solution quality  (n={N}, seed={SEED})")
for bar, gap in zip(bars[1:], [gap_ga_vs_ref, gap_ne_vs_ref]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{gap:+.1f}%", ha="center", va="bottom", fontsize=9)

axes[1].bar(["GA", "Neuroevolution"], [ga_opt_time, ne_opt_time],
            color=["#2196F3", "#FF5722"])
axes[1].set_ylabel("Optimization time  (s)")
axes[1].set_title("Wall-clock training time")

plt.suptitle(f"E4 Head-to-Head  —  n={N} DTSP, seed={SEED}", fontsize=12, y=1.02)
plt.tight_layout()

png_path = FIGURES / "e4_head_to_head.png"
pdf_path = FIGURES / "e4_head_to_head.pdf"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

wandb.log({
    "ref_cost": ref_cost,
    "ga_cost": ga_cost, "ne_cost": ne_cost,
    "gap_ne_vs_ga_pct": gap_ne_vs_ga,
    "gap_ga_vs_ref_pct": gap_ga_vs_ref,
    "gap_ne_vs_ref_pct": gap_ne_vs_ref,
    "ga_opt_time_s": ga_opt_time, "ne_opt_time_s": ne_opt_time,
    "summary": wandb.Image(str(png_path)),
    "ga_tour_gif": wandb.Video(str(FIGURES / "e4_ga_tour.gif"), fps=4, format="gif"),
    "ne_tour_gif": wandb.Video(str(FIGURES / "e4_ne_tour.gif"), fps=4, format="gif"),
})
wandb.finish()

# ── Console summary ───────────────────────────────────────────────────────────
W = 62
print(f"\n{'':=<{W}}")
print(f"{'Method':<22} {'Cost':>8} {'Gap vs ref':>12} {'Opt time':>10} {'Inf time':>9}")
print(f"{'-'*W}")
print(f"{'NN+2-opt (reference)':<22} {ref_cost:>8.4f} {'—':>12} {'—':>10} {'—':>9}")
print(f"{'GA':<22} {ga_cost:>8.4f} {gap_ga_vs_ref:>+11.2f}% {ga_opt_time:>9.1f}s {ga_inf_time*1e3:>8.3f}ms")
print(f"{'Neuroevolution':<22} {ne_cost:>8.4f} {gap_ne_vs_ref:>+11.2f}% {ne_opt_time:>9.1f}s {ne_inf_time*1e3:>8.3f}ms")
print(f"{'NE gap vs GA':<22} {gap_ne_vs_ga:>+21.2f}%")
print(f"{'':=<{W}}")
print(f"\nFigure → {pdf_path}")
print(f"Results → {RESULTS / 'e4_head_to_head.csv'}")
