import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TDTSPEnv(gym.Env):
    """
    Time-Dependent TSP Gymnasium environment.
    Set perturbation_fraction=0.0 for a static TSP (no dynamics).
    """

    def __init__(self, n_cities=12, seed=42, spike_factor=2.0, perturbation_fraction=0.2):
        self.n = n_cities
        self.spike_factor = spike_factor
        self.pert_frac = perturbation_fraction

        rng = np.random.default_rng(seed)
        self.coords = rng.uniform(0, 1, (n_cities, 2)).astype(np.float32)
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self._base_cost = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
        self._spike_seed = seed + 1000

        # obs: all city coords (2n) + visited mask (n) + current city coords (2) + normalised time (1)
        obs_dim = 3 * n_cities + 3
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_cities)

    def get_cost_matrix(self, t):
        if self.pert_frac == 0.0:
            return self._base_cost
        rng = np.random.default_rng(self._spike_seed + t)
        cost = self._base_cost.copy()
        rows, cols = np.triu_indices(self.n, k=1)
        n_perturb = max(1, int(len(rows) * self.pert_frac))
        chosen = rng.choice(len(rows), n_perturb, replace=False)
        for c in chosen:
            i, j = rows[c], cols[c]
            cost[i, j] *= self.spike_factor
            cost[j, i] *= self.spike_factor
        return cost

    def _obs(self):
        return np.concatenate([
            self.coords.flatten(),
            self.visited.astype(np.float32),
            self.coords[self.current],
            [self.t / self.n],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.t = 0
        self.current = 0
        self.visited = np.zeros(self.n, dtype=bool)
        self.visited[0] = True
        self.tour = [0]
        self.total_cost = 0.0
        return self._obs(), {}

    def step(self, action):
        cost_mat = self.get_cost_matrix(self.t)
        step_cost = float(cost_mat[self.current, action])
        self.total_cost += step_cost
        self.visited[action] = True
        self.current = action
        self.tour.append(action)
        self.t += 1

        done = bool(self.visited.all())
        if done:
            return_cost = float(self.get_cost_matrix(self.t)[self.current, 0])
            self.total_cost += return_cost
            self.tour.append(0)
            reward = -(step_cost + return_cost)
        else:
            reward = -step_cost

        return self._obs(), float(reward), done, False, {}

    def evaluate_tour(self, tour, start_t=0):
        cost, t = 0.0, start_t
        for i in range(len(tour) - 1):
            cost += float(self.get_cost_matrix(t)[tour[i], tour[i + 1]])
            t += 1
        return cost
