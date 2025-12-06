"""
blindness_mappo_vmas.py

MAPPO-style training on VMAS with optional blindness:
- Scenario: balance or wheel
- Parameter sharing, centralized critic, shared reward
- Optional "blind events": at random times, one random agent's observations
  are replaced by zeros for k steps; its actions are still applied.

Examples:
    # Baseline (no blindness)
    python blindness_mappo_vmas.py --scenario balance --blind_prob 0.0

    # Blindness: p=0.05, k=10, no self flag
    python blindness_mappo_vmas.py --scenario balance --blind_prob 0.05 --blind_k 10

    # Blindness + self-awareness flag
    python blindness_mappo_vmas.py --scenario balance --blind_prob 0.05 --blind_k 10 --blind_self_flag
"""

import argparse
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import vmas

# Utilities / Hyperparams
@dataclass
class Config:
    scenario: str = "balance"   # "balance" or "wheel"
    device: str = "cpu"
    num_envs: int = 1

    # Training budget
    total_steps: int = 1_000_000

    # PPO and GAE
    rollout_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    ppo_clip: float = 0.2

    # Optimizer and loss weights
    lr: float = 1e-4
    value_coeff: float = 0.5
    entropy_coeff: float = 0.001

    max_grad_norm: float = 0.5
    hidden_size: int = 128
    max_episode_steps: int = 300    # VMAS max_steps

    # Blindness parameters
    blind_prob: float = 0.0         # per-step probability to start a blind event
    blind_k: int = 0                # length of blind event (steps)
    blind_mask: str = "zeros"       # "zeros" or "noise"
    blind_self_flag: bool = False   # append 1-bit "I am blind" flag to obs


def mlp(input_dim, output_dim, hidden_size):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
    )

class GaussianPolicy(nn.Module):
    # Shared Gaussian policy for all agents (parameter sharing)
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden_size)
        # Log std is a learned parameter (shared across all agents)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        """
        obs: [batch, obs_dim]
        returns: action, log_prob, mean
        """
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, mean

    def evaluate_actions(self, obs, actions):
        """
        Used during PPO update.
        obs: [batch, obs_dim]
        actions: [batch, act_dim]
        """
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized value function that sees concatenated observations
    of all agents (global state proxy).
    """

    def __init__(self, global_obs_dim, hidden_size=128):
        super().__init__()
        self.net = mlp(global_obs_dim, 1, hidden_size)

    def forward(self, global_obs):
        """
        global_obs: [batch, global_obs_dim]
        returns: [batch]
        """
        v = self.net(global_obs)
        return v.squeeze(-1)

# Blindness Wrapper
class BlindnessWrapper:
    """
    Wraps a VMAS env and injects blindness:
    - At random steps, with probability blind_prob, if no one is currently blind,
      choose one random agent and blind it for blind_k steps.
    - Blindness means: that agent's observation is replaced (zeros or noise),
      but its actions are still applied normally.
    - At most one blind agent at a time.
    - Optional self_flag: append a 1-dim "I am blind" flag to each agent obs.
    """

    def __init__(self, env, blind_prob: float, blind_k: int, mask: str = "zeros", self_flag: bool = False):
        self.env = env
        self.blind_prob = blind_prob
        self.blind_k = blind_k
        self.mask = mask
        self.self_flag = self_flag

        self.n_agents = len(env.agents)
        self.device = env.device
        self.num_envs = env.num_envs  # expected 1

        # Track which agent is blind and how many steps remain
        self.current_blind_agent = None
        self.remaining_steps = 0

    def reset(self):
        obs_list = self.env.reset()
        self.current_blind_agent = None
        self.remaining_steps = 0
        # At reset there is no blindness yet, but we may append flags
        obs_list = self._append_flag(obs_list)
        return obs_list

    def _maybe_start_blind_event(self):
        # Possibly start a new blind event if none is active
        if self.current_blind_agent is not None:
            return
        if self.blind_prob <= 0.0 or self.blind_k <= 0:
            return
        if np.random.rand() < self.blind_prob:
            self.current_blind_agent = np.random.randint(0, self.n_agents)
            self.remaining_steps = self.blind_k

    def _apply_blindness(self, obs_list):
        # Modify obs_list in-place for the current blind agent (if any)
        if self.current_blind_agent is None or self.remaining_steps <= 0:
            return obs_list

        idx = self.current_blind_agent
        if self.mask == "zeros":
            obs_list[idx] = torch.zeros_like(obs_list[idx])
        elif self.mask == "noise":
            obs_list[idx] = torch.randn_like(obs_list[idx])
        else:
            obs_list[idx] = torch.zeros_like(obs_list[idx])

        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            self.current_blind_agent = None

        return obs_list

    def _append_flag(self, obs_list):
        # Append a 1-dim flag 'I am blind' to each agent's obs if enabled
        if not self.self_flag:
            return obs_list

        out = []
        for i, o in enumerate(obs_list):
            # o: [num_envs, obs_dim], num_envs = 1
            is_blind = (
                self.current_blind_agent is not None
                and i == self.current_blind_agent
                and self.remaining_steps > 0
            )
            flag_val = 1.0 if is_blind else 0.0
            flag = torch.full(
                (o.shape[0], 1),
                flag_val,
                dtype=o.dtype,
                device=o.device,
            )
            out.append(torch.cat([o, flag], dim=-1))
        return out

    def step(self, actions):
        # Step underlying env
        obs_list, rewards_list, done, info = self.env.step(actions)

        # If episode done, clear blindness and append flags (all zero)
        if done[0].item() == 1:
            self.current_blind_agent = None
            self.remaining_steps = 0
            obs_list = self._append_flag(obs_list)
            return obs_list, rewards_list, done, info

        # Possibly start a blind event
        self._maybe_start_blind_event()

        # Apply blindness if active
        if self.current_blind_agent is not None:
            obs_list = self._apply_blindness(obs_list)

        # Append self-blind flags (or pass-through if disabled)
        obs_list = self._append_flag(obs_list)

        return obs_list, rewards_list, done, info

    # Convenience so training code can still access env attributes
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def agents(self):
        return self.env.agents


# PPO and MAPPO helper functions
def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """
    rewards: [T]
    dones: [T]
    values: [T]
    next_value: scalar
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


# Training Loop
def train(config: Config):
    # Seeding for reproducibility (Comment out later)
    seed = 67
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device(config.device)

    # Creates base VMAS environment
    base_env = vmas.make_env(
        scenario=config.scenario,
        num_envs=config.num_envs,
        device=device,
        continuous_actions=True,
        max_steps=config.max_episode_steps,
        seed=0,
    )

    # Optionally wrap with blindness
    if config.blind_prob > 0.0 and config.blind_k > 0:
        env = BlindnessWrapper(
            base_env,
            blind_prob=config.blind_prob,
            blind_k=config.blind_k,
            mask=config.blind_mask,
            self_flag=config.blind_self_flag,
        )
        print(
            f"Blindness enabled: p={config.blind_prob}, k={config.blind_k}, "
            f"mask={config.blind_mask}, self_flag={config.blind_self_flag}"
        )
    else:
        env = base_env
        print("Blindness disabled (baseline).")

    # Get action bounds (assumed same for all agents)
    act_low = torch.as_tensor(env.action_space[0].low, device=device)
    act_high = torch.as_tensor(env.action_space[0].high, device=device)

    # Determine dimensions
    obs_sample = env.reset()
    n_agents = len(env.agents)
    obs_dim = obs_sample[0].shape[-1]
    act_dim = env.action_space[0].shape[0]

    print(f"Scenario: {config.scenario}")
    print(f"Num agents: {n_agents}, obs_dim: {obs_dim}, act_dim: {act_dim}")
    print(f"Total steps: {config.total_steps}")

    policy = GaussianPolicy(obs_dim, act_dim, hidden_size=config.hidden_size).to(device)
    critic = CentralizedCritic(
        global_obs_dim=n_agents * obs_dim,
        hidden_size=config.hidden_size,
    ).to(device)

    optimizer = optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=config.lr
    )

    # Buffers for one rollout (time x agents)
    T = config.rollout_length
    obs_buf = torch.zeros(T, n_agents, obs_dim, device=device)
    act_buf = torch.zeros(T, n_agents, act_dim, device=device)
    logp_buf = torch.zeros(T, n_agents, device=device)
    rew_buf = torch.zeros(T, device=device)     # shared team reward
    done_buf = torch.zeros(T, device=device)
    val_buf = torch.zeros(T, device=device)

    global_step = 0
    episode_returns = []
    episode_return = 0.0
    learning_curve = []  # (step, mean_return_last_10)

    # Initial observation (env already reset above)
    obs_list = obs_sample
    obs = [o[0] for o in obs_list]  # list of tensors [obs_dim]

    while global_step < config.total_steps:
        # Collect rollout
        for t in range(T):
            obs_tensor = torch.stack(obs, dim=0)  # [n_agents, obs_dim]
            obs_buf[t] = obs_tensor

            with torch.no_grad():
                actions, log_probs, _ = policy(obs_tensor)

                # Clip actions to env bounds
                actions = torch.max(torch.min(actions, act_high), act_low)

                # Centralized critic uses concatenated obs
                global_obs = obs_tensor.view(1, -1)  # [1, n_agents*obs_dim]
                values = critic(global_obs)

            val_buf[t] = values
            logp_buf[t] = log_probs
            act_buf[t] = actions

            vmas_actions = [a.unsqueeze(0) for a in actions]

            next_obs_list, rewards_list, done, info = env.step(vmas_actions)

            rewards_tensor = torch.stack([r[0] for r in rewards_list])  # [n_agents]
            shared_reward = rewards_tensor.mean()

            rew_buf[t] = shared_reward
            done_buf[t] = done[0]

            episode_return += shared_reward.item()
            global_step += 1

            obs = [o[0] for o in next_obs_list]

            if done[0].item() == 1:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs_list = env.reset()
                obs = [o[0] for o in obs_list]

            if global_step >= config.total_steps:
                break

        # Bootstrap value
        with torch.no_grad():
            obs_tensor = torch.stack(obs, dim=0)
            global_obs = obs_tensor.view(1, -1)
            next_value = critic(global_obs)

        # Compute GAE / returns
        advantages, returns = compute_gae(
            rewards=rew_buf,
            dones=done_buf,
            values=val_buf,
            next_value=next_value,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )

        adv_all = advantages.unsqueeze(1).expand(-1, n_agents)     # [T, n_agents]
        ret_all = returns.unsqueeze(1).expand(-1, n_agents)        # [T, n_agents]

        adv_mean = adv_all.mean()
        adv_std = adv_all.std() + 1e-8
        adv_all = (adv_all - adv_mean) / adv_std

        Tn = T * n_agents
        flat_obs = obs_buf.reshape(Tn, obs_dim)
        flat_act = act_buf.reshape(Tn, act_dim)
        flat_logp_old = logp_buf.reshape(Tn)
        flat_adv = adv_all.reshape(Tn)
        flat_ret = ret_all.reshape(Tn)

        global_obs_buf = obs_buf.reshape(T, n_agents * obs_dim)
        global_obs_flat = global_obs_buf.unsqueeze(1).expand(-1, n_agents, -1)
        global_obs_flat = global_obs_flat.contiguous().view(Tn, n_agents * obs_dim)

        # PPO Update
        for _ in range(config.ppo_epochs):
            new_logp, entropy = policy.evaluate_actions(flat_obs, flat_act)
            ratio = torch.exp(new_logp - flat_logp_old)

            surr1 = ratio * flat_adv
            surr2 = torch.clamp(
                ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip
            ) * flat_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            values_pred = critic(global_obs_flat)
            value_loss = (values_pred - flat_ret).pow(2).mean()

            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + config.value_coeff * value_loss
                + config.entropy_coeff * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()

        # Logging and learning curve
        if len(episode_returns) > 0 and global_step % (config.rollout_length * 4) < T:
            overall = float(np.mean(episode_returns))

            print(
                f"Step: {global_step} / {config.total_steps} | "
                f"Overall mean return: {overall:.3f}"
            )

            # Record for learning curve plotting
            learning_curve.append((global_step, overall))

    print("Training finished.")
    if len(episode_returns) > 0:
        final_mean = float(np.mean(episode_returns))
        print(
            f"Final mean episodic return over all episodes: "
            f"{final_mean:.3f}"
        )
    else:
        final_mean = 0.0
        print("No episodes finished, final mean set to 0.0")

    # Return final performance and the learning curve for plotting
    return final_mean, learning_curve


# CLI entry
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="balance",
        choices=["balance", "wheel"],
        help="VMAS scenario to train on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=500_000,
        help="Total environment steps.",
    )
    parser.add_argument(
        "--blind_prob",
        type=float,
        default=0.0,
        help="Per-step probability to start a blind event (0 = no blindness).",
    )
    parser.add_argument(
        "--blind_k",
        type=int,
        default=0,
        help="Duration (in steps) of a blind event.",
    )
    parser.add_argument(
        "--blind_mask",
        type=str,
        default="zeros",
        choices=["zeros", "noise"],
        help="How to replace blinded observations.",
    )
    parser.add_argument(
        "--blind_self_flag",
        action="store_true",
        help="Append a 1-bit 'I am blind' flag to each agent's observation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        scenario=args.scenario,
        device=args.device,
        total_steps=args.total_steps,
        blind_prob=args.blind_prob,
        blind_k=args.blind_k,
        blind_mask=args.blind_mask,
        blind_self_flag=args.blind_self_flag,
    )
    # We ignore the returned values when running directly from CLI
    train(cfg)