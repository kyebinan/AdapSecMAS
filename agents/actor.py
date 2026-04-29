# agents/actor.py
# =============================================================================
# AdapSecMAS — Actor network
# GRU-based recurrent policy for partial observability.
# Handles history h_t^i without explicit belief state.
# Parameter sharing: all 20 agents use the same weights.
#
# Input  : o_t^i ∈ R^12  (7D own obs + 5D agg_comm)
# Output : π(a | h_t^i)  — categorical distribution over 7 actions
#
# SRP: only responsible for action distribution computation.
#      Training logic lives in mappo.py.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from core.constants import DIM_OBS_TOTAL, N_ACTIONS, HIDDEN_SIZE


class Actor(nn.Module):
    """
    Recurrent actor network — shared across all agents (parameter sharing).

    Architecture:
        Linear(DIM_OBS_TOTAL → HIDDEN_SIZE)
        ReLU
        GRU(HIDDEN_SIZE → HIDDEN_SIZE)
        Linear(HIDDEN_SIZE → N_ACTIONS)
        Softmax

    The GRU hidden state h_t^i encodes the agent's history,
    enabling decision-making under partial observability.
    """

    def __init__(
        self,
        obs_dim    : int = DIM_OBS_TOTAL,
        action_dim : int = N_ACTIONS,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()
        self._hidden_size = hidden_size

        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            batch_first = True,
        )
        self.output_layer = nn.Linear(hidden_size, action_dim)

    def forward(
        self,
        obs    : torch.Tensor,          # (batch, obs_dim)
        hidden : torch.Tensor | None,   # (1, batch, hidden_size)
        action_mask: torch.Tensor | None = None,  # (batch, action_dim) bool
    ) -> tuple[Categorical, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        obs         : current observation
        hidden      : GRU hidden state from previous step
        action_mask : True = action available — masks unavailable actions

        Returns
        -------
        dist   : Categorical distribution over actions
        hidden : updated GRU hidden state
        """
        x = self.input_layer(obs)                   # (batch, hidden)
        x = x.unsqueeze(1)                          # (batch, 1, hidden) for GRU
        x, hidden = self.gru(x, hidden)             # (batch, 1, hidden)
        x = x.squeeze(1)                            # (batch, hidden)
        logits = self.output_layer(x)               # (batch, action_dim)

        if action_mask is not None:
            # Mask unavailable actions with large negative value
            logits = logits.masked_fill(~action_mask, float('-1e9'))

        dist = Categorical(logits=logits)
        return dist, hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return zero hidden state for episode start."""
        return torch.zeros(1, batch_size, self._hidden_size)