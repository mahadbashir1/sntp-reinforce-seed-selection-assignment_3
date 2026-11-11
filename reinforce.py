import random
import time

import numpy as np
import torch

from models import masked_distribution


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(env, policy, diffusion_seed):
    # sample 5 actions from the masked policy and keep their log-probabilities
    features, mask = env.reset(diffusion_seed)
    log_probs = []
    done = False
    while not done:
        logits = policy(torch.from_numpy(features))
        dist = masked_distribution(logits, torch.from_numpy(mask))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        features, mask, reward, done, info = env.step(action.item())
    return torch.stack(log_probs).sum(), reward, info["spread"]


def train(policy_class, env, seed, episodes=1000, batch_size=20, lr=0.001, max_grad_norm=1.0):
    set_seed(seed)
    policy = policy_class()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0
    history = []
    all_finite = True

    start = time.perf_counter()
    for batch_start in range(0, episodes, batch_size):
        log_prob_sums = []
        rewards = []
        for episode in range(batch_start, batch_start + batch_size):
            diffusion_seed = 1000000 + 10000 * seed + episode
            log_prob_sum, reward, spread = run_episode(env, policy, diffusion_seed)
            log_prob_sums.append(log_prob_sum)
            rewards.append(reward)
            history.append((episode, spread, reward))

        # L = -1/20 * sum_i (R_i - b) * sum_t log pi(a_t | s_t)
        # returns and baseline are plain numbers, so they get no gradient
        returns = torch.tensor(rewards, dtype=torch.float32)
        loss = -((returns - baseline) * torch.stack(log_prob_sums)).mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            all_finite = False
        optimizer.step()

        # baseline is updated only after the optimizer step
        baseline = 0.9 * baseline + 0.1 * float(np.mean(rewards))
    training_seconds = time.perf_counter() - start

    return policy, history, training_seconds, all_finite
