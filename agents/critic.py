# agents/critic.py
# =============================================================================
# AdapSecMAS — Centralised Critic network (CTDE)
# Used ONLY during training — not deployed at execution.
# Sees the concatenated observations of all agents (global context).
#
# Input  : concat(o_t^i, i=1..N) ∈ R^(N × DIM_OBS_TOTAL)
# Output : V(h_t, s_t) — scalar state value
#
# CTDE: centralized training, decentralized execution.
#   Training  : critic sees joint obs → sharp advantage estimate
#   Execution : only Actor(o_t^i) deployed on each agent
#
# SRP: only responsible for value estimation.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from core.constants import N_AGENTS, DIM_OBS_TOTAL


class Critic(nn.Module):
    """
    Centralised value network for MAPPO.

    Architecture:
        Linear(N × DIM_OBS_TOTAL → 256)
        ReLU
        Linear(256 → 128)
        ReLU
        Linear(128 → 1)

    Input is the concatenation of all agents' observations.
    This gives the critic access to global context during training,
    enabling low-variance advantage estimates.
    """

    def __init__(
        self,
        n_agents    : int = N_AGENTS,
        obs_dim     : int = DIM_OBS_TOTAL,
        hidden_size : int = 256,
    ):
        super().__init__()
        input_dim = n_agents * obs_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        joint_obs : (batch, N × DIM_OBS_TOTAL) — all agents concatenated

        Returns
        -------
        value : (batch, 1)
        """
        return self.net(joint_obs)