import time
import numpy as np


def _ox_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1)
    child[a : b + 1] = p1[a : b + 1]
    used = set(p1[a : b + 1])
    fill = [x for x in p2 if x not in used]
    ptr = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[ptr]
            ptr += 1
    return child


def _inversion_mutate(perm, rate):
    if np.random.random() < rate:
        a, b = sorted(np.random.choice(len(perm), 2, replace=False))
        perm[a : b + 1] = perm[a : b + 1][::-1]
    return perm


def _tournament(pop, fitness, k):
    idxs = np.random.choice(len(pop), k, replace=False)
    return pop[idxs[fitness[idxs].argmax()]].copy()


def _eval_pop(pop, cost_mat):
    # Vectorised: build all tours as (pop_size, n+1) and index cost_mat in one shot
    depot = np.zeros((len(pop), 1), dtype=int)
    tours = np.concatenate([depot, pop, depot], axis=1)   # (P, n+1)
    costs = cost_mat[tours[:, :-1], tours[:, 1:]].sum(axis=1)
    return -costs  # fitness = -cost (maximise)


def run_ga(env, pop_size=200, mutation_rate=0.1, n_generations=500, tournament_k=3, seed=0,
           gif_path=None, wandb_log=True):
    np.random.seed(seed)
    cities = np.arange(1, env.n)
    pop = np.array([np.random.permutation(cities) for _ in range(pop_size)])
    history = []
    start = time.time()

    for gen in range(n_generations):
        cost_mat = env.get_cost_matrix(gen)
        fitness = _eval_pop(pop, cost_mat)
        history.append({"generation": gen, "best_cost": float(-fitness.max())})

        elite = pop[fitness.argmax()].copy()
        new_pop = [elite]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fitness, tournament_k)
            p2 = _tournament(pop, fitness, tournament_k)
            child = _ox_crossover(p1, p2)
            child = _inversion_mutate(child, mutation_rate)
            new_pop.append(child)
        pop = np.array(new_pop)

    cost_mat = env.get_cost_matrix(n_generations)
    fitness = _eval_pop(pop, cost_mat)
    best_tour = [0] + list(pop[fitness.argmax()]) + [0]
    eval_cost = env.evaluate_tour(best_tour, start_t=0)

    if gif_path is not None:
        import wandb as _wandb
        from utils.visualization import create_tour_gif
        create_tour_gif(
            env, best_tour, gif_path,
            title=f"GA — cost={eval_cost:.3f}",
        )
        if wandb_log:
            _wandb.log({"best_tour_gif": _wandb.Video(str(gif_path), fps=4, format="gif")})

    return {
        "best_cost": float(-fitness.max()),
        "eval_cost": eval_cost,
        "best_tour": best_tour,
        "history": history,
        "runtime": time.time() - start,
    }
