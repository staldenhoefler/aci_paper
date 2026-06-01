import time
import numpy as np
import torch
import wandb


# --- weight-vector helpers ---------------------------------------------------

def _get_weights(policy):
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in policy.parameters()])


def _set_weights(policy, weights):
    idx = 0
    with torch.no_grad():
        for p in policy.parameters():
            size = p.numel()
            p.copy_(torch.FloatTensor(weights[idx : idx + size].reshape(p.shape)))
            idx += size


def evaluate_greedy(env, policy):
    """Public: one greedy rollout with current policy weights. Returns tour cost."""
    return _evaluate(env, policy)


# --- GA operators ------------------------------------------------------------

def _greedy_rollout(env, policy):
    """Greedy single-tour. Returns (tour list, cost)."""
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            dist = policy(torch.FloatTensor(obs), torch.BoolTensor(env.visited))
        obs, _, done, _, _ = env.step(dist.probs.argmax().item())
    return env.tour, env.total_cost


def _evaluate(env, policy):
    _, cost = _greedy_rollout(env, policy)
    return cost


def _tournament(pop, fitness, k):
    idxs = np.random.choice(len(pop), k, replace=False)
    return pop[idxs[fitness[idxs].argmax()]].copy()


def _uniform_crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2)


# --- main entry point --------------------------------------------------------

def run_neuroevolution(
    env,
    policy,
    pop_size=50,
    sigma=0.1,
    n_generations=500,
    elite_frac=0.2,
    tournament_k=3,
    crossover=True,
    seed=0,
    val_every=10,
    wandb_log=True,
    gif_path=None,
):
    """
    Optimise *policy* weights with a genetic algorithm (Such et al. 2018).
    No gradient computation — fitness is greedy tour cost.
    Modifies policy in-place; returns it with best weights loaded.
    """
    np.random.seed(seed)
    n_elite = max(1, int(pop_size * elite_frac))

    # Initialise population with diverse random perturbations around PyTorch init
    base = _get_weights(policy)
    n_params = len(base)
    pop = np.array([base + sigma * np.random.randn(n_params) for _ in range(pop_size)])

    history = []
    start = time.time()

    for gen in range(n_generations):
        # ---- evaluate every individual ----
        fitness = np.empty(pop_size)
        for i, w in enumerate(pop):
            _set_weights(policy, w)
            fitness[i] = -_evaluate(env, policy)   # maximise → minimise cost

        best_cost = -fitness.max()

        if gen % val_every == 0:
            history.append({"generation": gen, "best_cost": best_cost})
            if wandb_log:
                wandb.log({"best_cost": best_cost, "generation": gen})

        # ---- next generation ----
        elite_idxs = fitness.argsort()[-n_elite:]
        new_pop = list(pop[elite_idxs])             # elites survive unchanged

        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fitness, tournament_k)
            if crossover:
                p2 = _tournament(pop, fitness, tournament_k)
                child = _uniform_crossover(p1, p2)
            else:
                child = p1.copy()
            child += sigma * np.random.randn(n_params)
            new_pop.append(child)

        pop = np.array(new_pop)

    # ---- load best weights into policy (final population not yet evaluated) ----
    final_costs = []
    for w in pop:
        _set_weights(policy, w)
        final_costs.append(_evaluate(env, policy))
    best_idx = int(np.argmin(final_costs))
    _set_weights(policy, pop[best_idx])
    best_tour, final_cost = _greedy_rollout(env, policy)

    if gif_path is not None:
        from utils.visualization import create_tour_gif
        create_tour_gif(
            env, best_tour, gif_path,
            title=f"Neuroevolution — cost={final_cost:.3f}",
        )
        if wandb_log:
            wandb.log({"best_tour_gif": wandb.Video(str(gif_path), fps=4, format="gif")})

    return {
        "history": history,
        "final_cost": final_cost,
        "best_tour": best_tour,
        "runtime": time.time() - start,
    }
