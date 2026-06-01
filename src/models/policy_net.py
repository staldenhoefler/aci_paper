import torch
import torch.nn as nn

OBS_DIM = lambda n: 3 * n + 3  # coords(2n) + visited(n) + current_coords(2) + t(1)


class PolicyNet(nn.Module):
    def __init__(self, n_cities, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM(n_cities), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_cities),
        )

    def forward(self, obs, visited_mask):
        logits = self.net(obs)
        logits = logits.masked_fill(visited_mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits)


class ValueNet(nn.Module):
    """State-dependent baseline V(s) — estimates expected return from state s."""

    def __init__(self, n_cities, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM(n_cities), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
