from itertools import permutations
import numpy as np

# ── Baselines ─────────────────────────────────────────────────────────────────

def random_baseline(env, n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    cities = list(range(1, env.n))
    costs = [
        env.evaluate_tour([0] + rng.permutation(cities).tolist() + [0])
        for _ in range(n_samples)
    ]
    return float(np.mean(costs)), float(np.std(costs))


def brute_force_optimal(env, start_t=0):
    """Exact optimal via exhaustive search. Only feasible for n ≤ 12."""
    if env.n > 12:
        raise ValueError(f"brute_force_optimal is infeasible for n={env.n} (max 12).")
    cost_mat = env.get_cost_matrix(start_t)
    best_cost, best_tour = float("inf"), None
    for perm in permutations(range(1, env.n)):
        tour = [0] + list(perm) + [0]
        c = sum(cost_mat[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
        if c < best_cost:
            best_cost, best_tour = c, tour
    return best_tour, float(best_cost)


# ── Nearest-Neighbour + 2-opt (near-optimal for large n) ─────────────────────

def _nearest_neighbor(cost_mat, start=0):
    n = cost_mat.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    for _ in range(n - 1):
        curr = tour[-1]
        row = cost_mat[curr].copy()
        row[visited] = np.inf
        nxt = int(row.argmin())
        tour.append(nxt)
        visited[nxt] = True
    tour.append(start)
    return tour


def _two_opt(tour, cost_mat):
    """
    2-opt local search: repeatedly reverse sub-sequences that reduce tour cost.
    Considers all O(n²) edge pairs per pass; runs until no improvement is found.
    """
    best = list(tour)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                # Remove edges (i, i+1) and (j, j+1)
                # Add    edges (i, j)   and (i+1, j+1)
                delta = (
                    cost_mat[best[i],   best[j]]
                  + cost_mat[best[i+1], best[j+1]]
                  - cost_mat[best[i],   best[i+1]]
                  - cost_mat[best[j],   best[j+1]]
                )
                if delta < -1e-10:
                    best[i+1 : j+1] = best[i+1 : j+1][::-1]
                    improved = True
    return best


def reference_tour(env, start_t=0):
    """
    Compute a high-quality reference tour for benchmarking.

    Strategy:
      n ≤ 12  →  exact optimal (brute-force, O(n!))
      n > 12  →  nearest-neighbour initialisation + 2-opt local search
                 Typically reaches within 3–8 % of optimal for Euclidean TSP.

    Returns (tour, cost, method_name).
    """
    if env.n <= 12:
        tour, cost = brute_force_optimal(env, start_t=start_t)
        return tour, cost, "brute-force (exact)"

    cost_mat = env.get_cost_matrix(start_t)
    tour = _nearest_neighbor(cost_mat)
    #tour = _two_opt(tour, cost_mat)
    cost = float(sum(cost_mat[tour[i], tour[i+1]] for i in range(len(tour) - 1)))
    return tour, cost, "NN + 2-opt (near-optimal)"


def gap_to_reference(cost, ref_cost):
    """Percentage gap of *cost* relative to *ref_cost*."""
    return (cost - ref_cost) / ref_cost * 100
