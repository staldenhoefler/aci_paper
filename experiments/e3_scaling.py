"""
E3 — Scaling
Best GA config (from E1) vs Neuroevolution on sizes from params.yaml.
Records final tour cost and wall-clock time; produces grouped bar + log-time charts.
"""
import sys
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
from solvers.neuroevolution import run_neuroevolution
from utils.metrics import reference_tour, gap_to_reference
from utils.config import load_params, params_path, run_id

P = load_params()
RUN_ID = run_id()

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / RUN_ID
FIGURES = ROOT / "figures" / RUN_ID
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED         = P["env"]["seed"]
SIZES        = P["e3"]["sizes"]
GA_GENS      = P["e3"]["ga_n_generations"]
NE_POP       = P["neuroevolution"]["pop_size"]
NE_SIGMA     = P["neuroevolution"]["sigma"]
NE_GENS      = P["e3"]["ne_n_generations"]
ELITE_FRAC   = P["neuroevolution"]["elite_frac"]
TOURNAMENT_K = P["neuroevolution"]["tournament_k"]
CROSSOVER    = P["neuroevolution"]["crossover"]
HIDDEN       = P["neuroevolution"]["hidden_dim"]
NHEADS       = P["neuroevolution"]["n_heads"]
NLAYERS      = P["neuroevolution"]["n_layers"]
TRAIN_K      = P["neuroevolution"]["train_instances"]

# ── Load best GA config from E1 ───────────────────────────────────────────────
e1_csv = ROOT / "results" / RUN_ID / "e1_ga_sensitivity.csv"
if not e1_csv.exists():
    # Fall back to any existing E1 result
    candidates = sorted((ROOT / "results").glob("*/e1_ga_sensitivity.csv"), reverse=True)
    e1_csv = candidates[0] if candidates else None

if e1_csv and e1_csv.exists():
    e1_df = pd.read_csv(e1_csv)
    best_e1 = e1_df.loc[e1_df["eval_cost"].idxmin()]
    GA_POP = int(best_e1["pop_size"])
    GA_MUT = float(best_e1["mutation_rate"])
    print(f"Loaded best GA config from E1: pop={GA_POP}, mut={GA_MUT:.2f}")
else:
    print("WARNING: E1 results not found — using GA defaults from params.yaml")
    GA_POP = P["ga"]["pop_sizes"][1]   # middle value
    GA_MUT = P["ga"]["mutation_rates"][1]

print(f"\n=== E3: Scaling | sizes={SIZES}, GA_GENS={GA_GENS}, NE_GENS={NE_GENS} ===\n")
print(f"{'n':>4} {'GA cost':>10} {'GA time':>10} {'NE cost':>10} {'NE time':>10}")
print("-" * 48)

wandb.init(
    project=P["wandb"]["project"],
    name=f"e3-scaling-{RUN_ID}",
    config=dict(run_id=RUN_ID, sizes=SIZES, seed=SEED, ga_pop=GA_POP, ga_mut=GA_MUT,
                ga_gens=GA_GENS, ne_pop=NE_POP, ne_sigma=NE_SIGMA, ne_gens=NE_GENS,
                elite_frac=ELITE_FRAC, tournament_k=TOURNAMENT_K, crossover=CROSSOVER),
    mode=P["wandb"]["mode"],
)
wandb.save(str(params_path()), policy="now")
wb_table = wandb.Table(columns=["n", "ga_cost", "ga_time_s", "ne_cost", "ne_time_s"])

records = []
for n in SIZES:
    env = TDTSPEnv(n_cities=n, seed=SEED)

    _, ref_cost, _ = reference_tour(env)

    ga_res = run_ga(env, pop_size=GA_POP, mutation_rate=GA_MUT,
                    n_generations=GA_GENS, seed=SEED, wandb_log=False)

    torch.manual_seed(SEED)
    policy = AttentionPolicyNet(n_cities=n, hidden_dim=HIDDEN,
                                n_heads=NHEADS, n_layers=NLAYERS)
    train_envs = [TDTSPEnv(n_cities=n, seed=1000 + i) for i in range(TRAIN_K)]
    ne_res = run_neuroevolution(env, policy, pop_size=NE_POP, sigma=NE_SIGMA,
                                n_generations=NE_GENS, elite_frac=ELITE_FRAC,
                                tournament_k=TOURNAMENT_K, crossover=CROSSOVER,
                                seed=SEED, val_every=P["neuroevolution"]["val_every"],
                                wandb_log=False, train_envs=train_envs)

    row = dict(
        n=n,
        ref_cost=ref_cost,
        ga_cost=ga_res["eval_cost"],  ga_time_s=ga_res["runtime"],
        ga_gap=gap_to_reference(ga_res["eval_cost"], ref_cost),
        ne_cost=ne_res["final_cost"], ne_time_s=ne_res["runtime"],
        ne_gap=gap_to_reference(ne_res["final_cost"], ref_cost),
    )
    records.append(row)
    wb_table.add_data(n, row["ga_cost"], row["ga_time_s"], row["ne_cost"], row["ne_time_s"])
    print(f"{n:>4}  ref={ref_cost:.3f}  "
          f"GA={row['ga_cost']:.3f}({row['ga_gap']:+.1f}%)  "
          f"NE={row['ne_cost']:.3f}({row['ne_gap']:+.1f}%)")

df = pd.DataFrame(records)
df.to_csv(RESULTS / "e3_scaling.csv", index=False)

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(SIZES))
w = 0.35

axes[0].bar(x - w / 2, df["ga_cost"], w, label="GA", color="#2196F3")
axes[0].bar(x + w / 2, df["ne_cost"], w, label="Neuroevolution", color="#FF5722")
axes[0].plot(x, df["ref_cost"], "k--o", linewidth=1.5, markersize=5,
             label="NN+2-opt ref", zorder=5)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"n={n}" for n in SIZES])
axes[0].set_ylabel("Final tour cost")
axes[0].set_title("Solution quality vs. instance size")
axes[0].legend()

axes[1].plot(SIZES, df["ga_time_s"], "o-", color="#2196F3", label="GA", linewidth=2)
axes[1].plot(SIZES, df["ne_time_s"], "o-", color="#FF5722", label="Neuroevolution", linewidth=2)
axes[1].set_yscale("log")
axes[1].set_xlabel("Instance size  n")
axes[1].set_ylabel("Runtime  (s, log scale)")
axes[1].set_title("Wall-clock time vs. instance size")
axes[1].legend()

plt.tight_layout()
png_path = FIGURES / "e3_scaling.png"
pdf_path = FIGURES / "e3_scaling.pdf"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

wandb.log({"scaling_table": wb_table, "scaling_fig": wandb.Image(str(png_path))})
wandb.finish()

print(f"\nFigures saved to {pdf_path}")
print(f"Results saved to {RESULTS / 'e3_scaling.csv'}")
