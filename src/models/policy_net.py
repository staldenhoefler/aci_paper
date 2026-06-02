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


class AttentionPolicyNet(nn.Module):
    """
    Size-agnostic attention policy (Kool et al. 2019, simplified).
    Permutation-equivariant encoder + pointer-style decoder, so one set of
    weights runs on any instance size n. Parses the flat env obs internally,
    keeping the (obs, visited_mask) -> Categorical interface of PolicyNet.
    No dropout/batch-stats: forward is deterministic for a fixed weight vector
    (required for stable neuroevolution fitness).
    """

    NODE_FEAT = 6  # x, y, visited, current_x, current_y, t

    def __init__(self, n_cities=None, hidden_dim=32, n_heads=4, n_layers=1):
        super().__init__()
        self.d = hidden_dim
        self.embed = nn.Linear(self.NODE_FEAT, hidden_dim)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.0, batch_first=True)
            for _ in range(n_layers)
        )
        self.ff = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(),
                          nn.Linear(2 * hidden_dim, hidden_dim))
            for _ in range(n_layers)
        )
        self.norm1 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_layers))
        self.q_proj = nn.Linear(2 * hidden_dim + 1, hidden_dim)  # [graph, current-node, t]

    def _node_features(self, obs, n):
        coords = obs[..., : 2 * n].reshape(*obs.shape[:-1], n, 2)
        visited = obs[..., 2 * n : 3 * n].unsqueeze(-1)
        cur = obs[..., 3 * n : 3 * n + 2].unsqueeze(-2).expand_as(coords)
        t = obs[..., 3 * n + 2 : 3 * n + 3].unsqueeze(-2).expand(*coords.shape[:-1], 1)
        return torch.cat([coords, visited, cur, t], dim=-1)

    def forward(self, obs, visited_mask):
        single = obs.dim() == 1
        if single:
            obs, visited_mask = obs.unsqueeze(0), visited_mask.unsqueeze(0)
        n = visited_mask.shape[-1]

        h = self.embed(self._node_features(obs, n))            # (B, n, d)
        for attn, ff, n1, n2 in zip(self.attn, self.ff, self.norm1, self.norm2):
            a, _ = attn(h, h, h, need_weights=False)
            h = n1(h + a)
            h = n2(h + ff(h))

        coords = obs[..., : 2 * n].reshape(-1, n, 2)
        cur_coords = obs[..., 3 * n : 3 * n + 2]               # (B, 2)
        cur_idx = ((coords - cur_coords.unsqueeze(1)) ** 2).sum(-1).argmin(-1)  # (B,)
        cur_embed = h[torch.arange(h.shape[0]), cur_idx]       # (B, d)
        graph = h.mean(dim=1)                                  # (B, d)
        t = obs[..., 3 * n + 2 : 3 * n + 3]                    # (B, 1)

        q = self.q_proj(torch.cat([graph, cur_embed, t], dim=-1))   # (B, d)
        logits = (h * q.unsqueeze(1)).sum(-1) / (self.d ** 0.5)     # (B, n)
        logits = logits.masked_fill(visited_mask, float("-inf"))
        if single:
            logits = logits[0]
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
