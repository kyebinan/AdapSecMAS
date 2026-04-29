# agents/buffer.py
# =============================================================================
# AdapSecMAS — RolloutBuffer
# Stores one rollout of experience and computes GAE advantages.
# SRP: only responsible for data storage and advantage computation.
#      No network forward passes here.
# =============================================================================

from __future__ import annotations

import numpy as np
import torch

from core.constants import (
    N_AGENTS, DIM_OBS_TOTAL, N_ACTIONS,
    GAMMA, LAMBDA_GAE, UPDATE_EVERY,
)


class RolloutBuffer:
    """
    Fixed-size buffer for one PPO rollout.

    Stores per-step data for all agents:
      obs, actions, log_probs, rewards, values, dones, masks

    Computes GAE advantages after the rollout is complete.

    SRP: storage and GAE only — no network calls, no env interaction.
    """

    def __init__(
        self,
        n_agents    : int = N_AGENTS,
        obs_dim     : int = DIM_OBS_TOTAL,
        capacity    : int = UPDATE_EVERY,
        gamma       : float = GAMMA,
        lambda_gae  : float = LAMBDA_GAE,
    ):
        self._n         = n_agents
        self._obs_dim   = obs_dim
        self._capacity  = capacity
        self._gamma     = gamma
        self._lambda    = lambda_gae
        self._ptr       = 0
        self._full      = False

        # Pre-allocate buffers — shape (capacity, n_agents, ...)
        self.obs       = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((capacity, n_agents),           dtype=np.int64)
        self.log_probs = np.zeros((capacity, n_agents),           dtype=np.float32)
        self.rewards   = np.zeros((capacity,),                    dtype=np.float32)
        self.values    = np.zeros((capacity,),                    dtype=np.float32)
        self.dones     = np.zeros((capacity,),                    dtype=np.float32)
        self.masks     = np.ones( (capacity, n_agents, N_ACTIONS),dtype=np.float32)

    def add(
        self,
        obs      : dict[int, np.ndarray],   # {agent_id: obs}
        actions  : dict[int, int],
        log_probs: dict[int, float],
        reward   : float,
        value    : float,
        done     : bool,
        masks    : dict[int, list[bool]] | None = None,
    ) -> None:
        """Store one step of experience."""
        t = self._ptr

        for i in range(self._n):
            self.obs[t, i]       = obs[i]
            self.actions[t, i]   = actions[i]
            self.log_probs[t, i] = log_probs[i]
            if masks is not None:
                self.masks[t, i] = np.array(masks[i], dtype=np.float32)

        self.rewards[t] = reward
        self.values[t]  = value
        self.dones[t]   = float(done)

        self._ptr  = (self._ptr + 1) % self._capacity
        self._full = self._full or (self._ptr == 0)

    def compute_advantages(self, last_value: float) -> np.ndarray:
        """
        Generalised Advantage Estimation (GAE-λ).
        Returns advantages array of shape (capacity,).

        GAE: Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
             δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        Iterates backwards through the rollout.
        """
        size        = self._capacity
        advantages  = np.zeros(size, dtype=np.float32)
        last_gae    = 0.0
        next_value  = last_value
        next_done   = 0.0

        for t in reversed(range(size)):
            not_done   = 1.0 - self.dones[t]
            delta      = (
                self.rewards[t]
                + self._gamma * next_value * not_done
                - self.values[t]
            )
            last_gae   = delta + self._gamma * self._lambda * not_done * last_gae
            advantages[t] = last_gae

            next_value = self.values[t]
            next_done  = self.dones[t]

        return advantages

    def get_batches(
        self,
        last_value : float,
        batch_size : int,
        device     : torch.device,
    ):
        """
        Compute advantages and yield mini-batches for PPO update.
        Shuffles indices for each epoch.

        Yields
        ------
        obs_b       : (batch, n_agents, obs_dim)
        actions_b   : (batch, n_agents)
        log_probs_b : (batch, n_agents)
        advantages_b: (batch,)
        returns_b   : (batch,)
        masks_b     : (batch, n_agents, n_actions)
        """
        advantages = self.compute_advantages(last_value)
        returns    = advantages + self.values

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        size    = self._capacity
        indices = np.random.permutation(size)

        for start in range(0, size, batch_size):
            idx = indices[start : start + batch_size]

            yield (
                torch.FloatTensor(self.obs[idx]).to(device),
                torch.LongTensor(self.actions[idx]).to(device),
                torch.FloatTensor(self.log_probs[idx]).to(device),
                torch.FloatTensor(advantages[idx]).to(device),
                torch.FloatTensor(returns[idx]).to(device),
                torch.BoolTensor(self.masks[idx].astype(bool)).to(device),
            )

    def reset(self) -> None:
        """Clear buffer — called after each PPO update."""
        self._ptr  = 0
        self._full = False

    @property
    def is_ready(self) -> bool:
        """True when the buffer has collected enough steps for an update."""
        return self._full or self._ptr == 0 and self._full