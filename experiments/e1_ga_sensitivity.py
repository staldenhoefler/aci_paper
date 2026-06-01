"""
E1 — GA Sensitivity Analysis
3×3 grid search: pop_size × mutation_rate on the DTSP instance (params.yaml).
Records final tour cost and runtime per config; produces a cost heatmap.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from environment.tdtsp import TDTSPEnv
from solvers.ga import run_ga
from utils.metrics import reference_tour, gap_to_reference
from utils.config import load_params, params_path, run_id

P = load_params()
RUN_ID = run_id()

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / RUN_ID
FIGURES = ROOT / "figures" / RUN_ID
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

N            = P["e1"]["n_cities"]
SEED         = P["env"]["seed"]
N_GENS       = P["ga"]["n_generations"]
POP_SIZES    = P["ga"]["pop_sizes"]
MUTATION_RATES = P["ga"]["mutation_rates"]
TOURNAMENT_K = P["ga"]["tournament_k"]

env = TDTSPEnv(n_cities=N, seed=SEED)

print(f"=== E1: GA Sensitivity | n={N}, seed={SEED}, gens={N_GENS} ===\n")
print("Computing reference tour...")
ref_tour, ref_cost, ref_method = reference_tour(env)
print(f"Reference ({ref_method}): {ref_cost:.4f}\n")

print(f"{'pop_size':>10} {'mut_rate':>10} {'eval_cost':>12} {'gap_pct':>10} {'runtime':>10}")
print("-" * 56)

wandb.init(
    project=P["wandb"]["project"],
    name=f"e1-ga-sensitivity-n{N}-{RUN_ID}",
    config=dict(run_id=RUN_ID, n_cities=N, seed=SEED, n_generations=N_GENS,
                pop_sizes=POP_SIZES, mutation_rates=MUTATION_RATES,
                tournament_k=TOURNAMENT_K),
    mode=P["wandb"]["mode"],
)
wandb.save(str(params_path()), policy="now")
wb_table = wandb.Table(columns=["pop_size", "mutation_rate", "eval_cost", "gap_pct", "runtime_s"])
wandb.log({"ref_cost": ref_cost, "ref_method": ref_method})

records = []
best_so_far = {"cost": float("inf"), "pop_size": None, "mut_rate": None}

for pop_size in POP_SIZES:
    for mut_rate in MUTATION_RATES:
        result = run_ga(
            env,
            pop_size=pop_size,
            mutation_rate=mut_rate,
            n_generations=N_GENS,
            tournament_k=TOURNAMENT_K,
            seed=SEED,
            wandb_log=False,
        )
        if result["eval_cost"] < best_so_far["cost"]:
            best_so_far = {"cost": result["eval_cost"], "pop_size": pop_size, "mut_rate": mut_rate,
                           "best_tour": result["best_tour"]}
        cost = result["eval_cost"]
        gap  = gap_to_reference(cost, ref_cost)
        rt   = result["runtime"]
        print(f"{pop_size:>10} {mut_rate:>10.2f} {cost:>12.4f} {gap:>+9.2f}% {rt:>9.1f}s")
        records.append({"pop_size": pop_size, "mutation_rate": mut_rate,
                        "eval_cost": cost, "gap_pct": gap, "runtime_s": rt})
        wb_table.add_data(pop_size, mut_rate, cost, gap, rt)

        conv_df = pd.DataFrame(result["history"])
        conv_df.to_csv(
            RESULTS / f"e1_ga_p{pop_size}_m{int(mut_rate*100):02d}.csv", index=False
        )

df = pd.DataFrame(records)
df.to_csv(RESULTS / "e1_ga_sensitivity.csv", index=False)

# ── Heatmaps ─────────────────────────────────────────────────────────────────
pivot_cost = df.pivot(index="pop_size", columns="mutation_rate", values="eval_cost")
pivot_gap  = df.pivot(index="pop_size", columns="mutation_rate", values="gap_pct")
pivot_rt   = df.pivot(index="pop_size", columns="mutation_rate", values="runtime_s")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.heatmap(pivot_cost, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=axes[0], cbar_kws={"label": "Tour cost"})
axes[0].set_title(f"Final tour cost  (n={N}, {N_GENS} gens)")
axes[0].set_xlabel("Mutation rate")
axes[0].set_ylabel("Population size")

sns.heatmap(pivot_gap, annot=True, fmt=".1f", cmap="RdYlGn_r",
            ax=axes[1], cbar_kws={"label": "Gap to ref (%)"},
            annot_kws={"size": 9})
axes[1].set_title(f"Gap to NN+2-opt reference  (%)")
axes[1].set_xlabel("Mutation rate")
axes[1].set_ylabel("Population size")

sns.heatmap(pivot_rt, annot=True, fmt=".1f", cmap="Blues",
            ax=axes[2], cbar_kws={"label": "Runtime (s)"})
axes[2].set_title("Wall-clock runtime (s)")
axes[2].set_xlabel("Mutation rate")
axes[2].set_ylabel("Population size")

plt.tight_layout()
png_path = FIGURES / "e1_heatmap.png"
pdf_path = FIGURES / "e1_heatmap.pdf"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

# ── GIF of best tour ─────────────────────────────────────────────────────────
from utils.visualization import create_tour_gif
gif_path = FIGURES / "e1_best_tour.gif"
create_tour_gif(env, best_so_far["best_tour"], gif_path,
                title=f"GA best tour  (pop={best_so_far['pop_size']}, "
                      f"mut={best_so_far['mut_rate']:.2f}, cost={best_so_far['cost']:.3f})")

wandb.log({
    "results": wb_table,
    "heatmap": wandb.Image(str(png_path)),
    "best_tour_gif": wandb.Video(str(gif_path), fps=4, format="gif"),
})
wandb.finish()

# ── Summary ───────────────────────────────────────────────────────────────────
best = df.loc[df["eval_cost"].idxmin()]
print(f"\nReference ({ref_method}): {ref_cost:.4f}")
print(f"Best config → pop_size={int(best['pop_size'])}, "
      f"mutation_rate={best['mutation_rate']:.2f}, "
      f"cost={best['eval_cost']:.4f}  (gap={best['gap_pct']:+.2f}%)")
print(f"Figures saved to {FIGURES}/e1_heatmap.{{pdf,png}}")
print(f"Results saved to {RESULTS}/e1_ga_sensitivity.csv")
