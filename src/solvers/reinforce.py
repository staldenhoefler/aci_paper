import time
import numpy as np
import torch
import torch.nn.functional as F
import wandb


def run_episode(env, policy, greedy=False):
    obs, _ = env.reset()
    log_probs, rewards, obs_list, entropies = [], [], [], []
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs)
        obs_list.append(obs_t)
        mask = torch.BoolTensor(env.visited)
        dist = policy(obs_t, mask)
        if greedy:
            action = dist.probs.argmax(dim=-1).item()
        else:
            action = dist.sample().item()
            log_probs.append(dist.log_prob(torch.tensor(action)))
            entropies.append(dist.entropy())
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
    return log_probs, rewards, env.total_cost, obs_list, entropies


def _per_step_returns(rewards):
    """Undiscounted future return G_t from each timestep."""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G += r
        returns.insert(0, G)
    return torch.FloatTensor(returns)


def run_reinforce(
    env,
    policy,
    policy_optimizer,
    n_episodes=3000,
    val_every=100,
    batch_size=8,
    seed=1,
    value_net=None,
    value_optimizer=None,
    entropy_coef=0.01,
    wandb_log=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    history = []
    start = time.time()
    ep = 0

    while ep < n_episodes:
        policy_losses, value_losses = [], []
        batch_train_cost = 0.0

        for _ in range(batch_size):
            log_probs, rewards, total_cost, obs_list, entropies = run_episode(env, policy)
            G_t = _per_step_returns(rewards)  # (T,)
            batch_train_cost += total_cost
            ep += 1

            if value_net is not None:
                obs_batch = torch.stack(obs_list)           # (T, obs_dim)
                V = value_net(obs_batch)                    # (T,)
                advantages = G_t - V.detach()
                value_losses.append(F.mse_loss(V, G_t))
            else:
                advantages = G_t

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Entropy bonus: encourages exploration by penalising low-entropy policies
            entropy_bonus = torch.stack(entropies).mean()

            policy_losses.append(
                (-torch.stack(log_probs) * advantages).sum()
                - entropy_coef * entropy_bonus
            )

        # Policy update
        policy_optimizer.zero_grad()
        torch.stack(policy_losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_optimizer.step()

        # Value network update
        if value_net is not None and value_optimizer is not None and value_losses:
            value_optimizer.zero_grad()
            torch.stack(value_losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            value_optimizer.step()

        prev_ep = ep - batch_size
        if (ep // val_every) > (prev_ep // val_every):
            _, _, val_cost, _, _ = run_episode(env, policy, greedy=True)
            avg_train = batch_train_cost / batch_size
            entry = {"episode": ep, "val_cost": val_cost, "train_cost": avg_train}
            history.append(entry)
            if wandb_log:
                wandb.log({"val_cost": val_cost, "train_cost": avg_train, "episode": ep})

    _, _, final_cost, _, _ = run_episode(env, policy, greedy=True)
    return {"history": history, "final_cost": final_cost, "runtime": time.time() - start}
