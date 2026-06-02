"""
E2 — Neuroevolution Stability
Three independent seeds on the DTSP instance (params.yaml).
Produces a convergence curve with mean ± std band.
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
from solvers.neuroevolution import run_neuroevolution
from utils.config import load_params, params_path, run_id

P = load_params()
RUN_ID = run_id()

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / RUN_ID
FIGURES = ROOT / "figures" / RUN_ID
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

N            = P["e2"]["n_cities"]
SEED         = P["env"]["seed"]
SEEDS        = P["neuroevolution"]["seeds"]
NE_POP       = P["neuroevolution"]["pop_size"]
NE_SIGMA     = P["neuroevolution"]["sigma"]
NE_GENS      = P["neuroevolution"]["n_generations"]
NE_VAL_EVERY = P["neuroevolution"]["val_every"]
ELITE_FRAC   = P["neuroevolution"]["elite_frac"]
TOURNAMENT_K = P["neuroevolution"]["tournament_k"]
CROSSOVER    = P["neuroevolution"]["crossover"]
HIDDEN       = P["neuroevolution"]["hidden_dim"]
NHEADS       = P["neuroevolution"]["n_heads"]
NLAYERS      = P["neuroevolution"]["n_layers"]
TRAIN_K      = P["neuroevolution"]["train_instances"]

env = TDTSPEnv(n_cities=N, seed=SEED)
train_envs = [TDTSPEnv(n_cities=N, seed=1000 + i) for i in range(TRAIN_K)]

print(f"=== E2: Neuroevolution Stability | n={N}, env-seed={SEED} ===\n")
print(f"{'seed':>6} {'final_cost':>12} {'runtime':>10}")
print("-" * 32)

wandb.init(
    project=P["wandb"]["project"],
    name=f"e2-neuroevo-stability-n{N}-{RUN_ID}",
    config=dict(run_id=RUN_ID, n_cities=N, env_seed=SEED, ne_seeds=SEEDS,
                pop_size=NE_POP, sigma=NE_SIGMA, n_generations=NE_GENS,
                elite_frac=ELITE_FRAC, tournament_k=TOURNAMENT_K, crossover=CROSSOVER),
    mode=P["wandb"]["mode"],
)
wandb.save(str(params_path()), policy="now")

all_histories = []
summary_rows = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = AttentionPolicyNet(n_cities=N, hidden_dim=HIDDEN,
                                n_heads=NHEADS, n_layers=NLAYERS)

    gif_path = FIGURES / f"e2_neuroevo_seed{seed}.gif"
    result = run_neuroevolution(
        env, policy,
        pop_size=NE_POP,
        sigma=NE_SIGMA,
        n_generations=NE_GENS,
        elite_frac=ELITE_FRAC,
        tournament_k=TOURNAMENT_K,
        crossover=CROSSOVER,
        seed=seed,
        val_every=NE_VAL_EVERY,
        wandb_log=False,
        gif_path=gif_path,
        train_envs=train_envs,
    )

    print(f"{seed:>6} {result['final_cost']:>12.4f} {result['runtime']:>9.1f}s")
    df_seed = pd.DataFrame(result["history"]).assign(seed=seed)
    all_histories.append(df_seed)
    summary_rows.append({"seed": seed, "final_cost": result["final_cost"],
                         "runtime_s": result["runtime"]})

    pd.DataFrame(result["history"]).to_csv(
        RESULTS / f"e2_neuroevo_seed{seed}.csv", index=False
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS / "e2_neuroevo_stability.csv", index=False)

# ── Convergence figure ────────────────────────────────────────────────────────
merged = pd.concat(all_histories)
stats = (merged.groupby("generation")["best_cost"]
               .agg(["mean", "std"])
               .reset_index())

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2196F3", "#4CAF50", "#FF5722"]
for (seed, df_s), c in zip(merged.groupby("seed"), colors):
    ax.plot(df_s["generation"], df_s["best_cost"],
            color=c, alpha=0.5, linewidth=1, label=f"Seed {seed}")
ax.plot(stats["generation"], stats["mean"],
        "k-", linewidth=2, label="Mean")
ax.fill_between(stats["generation"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.15, color="gray", label="±1 std")
ax.set_xlabel("Generation")
ax.set_ylabel("Best tour cost")
ax.set_title(f"Neuroevolution convergence  (n={N}, {len(SEEDS)} seeds)")
ax.legend(fontsize=9)
plt.tight_layout()

png_path = FIGURES / "e2_convergence.png"
pdf_path = FIGURES / "e2_convergence.pdf"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

best_seed_row = summary_df.loc[summary_df["final_cost"].idxmin()]
best_gif = FIGURES / f"e2_neuroevo_seed{int(best_seed_row['seed'])}.gif"

wandb.log({
    "convergence": wandb.Image(str(png_path)),
    "best_tour_gif": wandb.Video(str(best_gif), fps=4, format="gif"),
})
wandb.finish()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nMean final cost : {summary_df['final_cost'].mean():.4f} "
      f"± {summary_df['final_cost'].std():.4f}")
print(f"Figure saved to {pdf_path}")
