# agents/mappo.py
# =============================================================================
# AdapSecMAS — MAPPOTrainer
# Multi-Agent PPO with centralised critic (CTDE) and parameter sharing.
# SRP: only responsible for the learning update.
#      Environment interaction lives in train.py.
#
# Algorithm:
#   - One shared Actor for all 20 agents (parameter sharing)
#   - One centralised Critic (sees joint obs — training only)
#   - PPO clip objective with GAE advantages
#   - Entropy annealing for exploration → exploitation
#   - Action masking via SecurityLevelMachine
# =============================================================================

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.actor  import Actor
from agents.critic import Critic
from agents.buffer import RolloutBuffer

from security.levels       import SecurityLevel, ACTION_MASKS
from security.state_machine import SecurityLevelMachine

from core.constants import (
    N_AGENTS, DIM_OBS_TOTAL, N_ACTIONS,
    GAMMA, LAMBDA_GAE, CLIP_EPS,
    LR_ACTOR, LR_CRITIC,
    UPDATE_EVERY, PPO_EPOCHS, BATCH_SIZE,
    HIDDEN_SIZE,
    ENT_START, ENT_END, ENT_ANNEAL_STEPS,
)


class MAPPOTrainer:
    """
    MAPPO trainer with centralised critic and parameter sharing.

    CTDE:
      Training  : Critic sees joint_obs = concat(o_t^i, i=1..N)
      Execution : Actor uses only o_t^i — decentralised

    Parameter sharing:
      All 20 agents share one Actor network.
      Each agent maintains its own GRU hidden state.

    Action masking:
      SecurityLevelMachine provides per-agent action masks.
      Unavailable actions are masked before sampling.
    """

    def __init__(
        self,
        n_agents    : int   = N_AGENTS,
        obs_dim     : int   = DIM_OBS_TOTAL,
        action_dim  : int   = N_ACTIONS,
        hidden_size : int   = HIDDEN_SIZE,
        lr_actor    : float = LR_ACTOR,
        lr_critic   : float = LR_CRITIC,
        clip_eps    : float = CLIP_EPS,
        ppo_epochs  : int   = PPO_EPOCHS,
        batch_size  : int   = BATCH_SIZE,
        update_every: int   = UPDATE_EVERY,
        device      : str   = "auto",
    ):
        self._n          = n_agents
        self._clip_eps   = clip_eps
        self._ppo_epochs = ppo_epochs
        self._batch_size = batch_size
        self._total_steps= 0

        # Device selection
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Networks
        self.actor  = Actor(obs_dim, action_dim, hidden_size).to(self._device)
        self.critic = Critic(n_agents, obs_dim).to(self._device)

        # Optimisers
        self._opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self._opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Buffer
        self.buffer = RolloutBuffer(
            n_agents   = n_agents,
            obs_dim    = obs_dim,
            capacity   = update_every,
            gamma      = GAMMA,
            lambda_gae = LAMBDA_GAE,
        )

        # GRU hidden states — one per agent
        self._hiddens: dict[int, torch.Tensor] = {}

        # Security Level Machines — one per agent
        self._slms: dict[int, SecurityLevelMachine] = {
            i: SecurityLevelMachine() for i in range(n_agents)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def reset_hiddens(self) -> None:
        """Reset GRU hidden states — called at episode start."""
        for i in range(self._n):
            self._hiddens[i] = self.actor.init_hidden(batch_size=1).to(self._device)
        for slm in self._slms.values():
            slm.reset()

    def act(
        self,
        obs: dict[int, np.ndarray],
    ) -> tuple[dict[int, int], dict[int, float], dict[int, list[bool]]]:
        """
        Sample actions for all agents from the shared Actor.
        Returns actions, log_probs, and action masks.

        Called at each environment step during rollout collection.
        """
        actions   : dict[int, int]        = {}
        log_probs : dict[int, float]      = {}
        masks     : dict[int, list[bool]] = {}

        self.actor.eval()
        with torch.no_grad():
            for i in range(self._n):
                obs_t = torch.FloatTensor(obs[i]).unsqueeze(0).to(self._device)

                # Get action mask from SLM
                mask_list = self._slms[i].action_mask()
                mask_t    = torch.BoolTensor(mask_list).unsqueeze(0).to(self._device)

                dist, new_hidden = self.actor(obs_t, self._hiddens[i], mask_t)
                action           = dist.sample()
                log_prob         = dist.log_prob(action)

                self._hiddens[i]  = new_hidden
                actions[i]        = int(action.item())
                log_probs[i]      = float(log_prob.item())
                masks[i]          = mask_list

                # Update SLM with current obs
                self._slms[i].update(obs[i][:7])

        return actions, log_probs, masks

    def get_value(self, obs: dict[int, np.ndarray]) -> float:
        """
        Compute state value from joint observation (centralised critic).
        Called at the end of each rollout for GAE bootstrap.
        """
        joint = self._build_joint_obs(obs)
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(joint)
        return float(value.item())

    def record(
        self,
        obs      : dict[int, np.ndarray],
        actions  : dict[int, int],
        log_probs: dict[int, float],
        reward   : float,
        done     : bool,
        masks    : dict[int, list[bool]],
    ) -> None:
        """Store one step of experience in the buffer."""
        joint = self._build_joint_obs(obs)
        self.critic.eval()
        with torch.no_grad():
            value = float(self.critic(joint).item())

        self.buffer.add(obs, actions, log_probs, reward, value, done, masks)

    def should_update(self) -> bool:
        """True when buffer is full and a PPO update should run."""
        return self.buffer._ptr == 0 and self.buffer._full

    def update(self, last_obs: dict[int, np.ndarray]) -> dict[str, float]:
        """
        Run PPO_EPOCHS passes of the PPO update over the collected rollout.
        Returns a dict of training metrics for logging.

        Steps:
          1. Compute last value for GAE bootstrap
          2. Get mini-batches from buffer (shuffled)
          3. Recompute log_probs and entropy under current policy
          4. Clip policy ratio
          5. Update actor and critic
          6. Reset buffer
        """
        last_value = self.get_value(last_obs)
        entropy_coef = self._entropy_coef()

        metrics = {
            "policy_loss": 0.0,
            "value_loss" : 0.0,
            "entropy"    : 0.0,
            "n_updates"  : 0,
        }

        self.actor.train()
        self.critic.train()

        for _ in range(self._ppo_epochs):
            for batch in self.buffer.get_batches(last_value, self._batch_size, self._device):
                obs_b, actions_b, old_log_probs_b, advantages_b, returns_b, masks_b = batch

                # -- Recompute log_probs for all agents in batch --
                batch_size = obs_b.shape[0]
                all_log_probs = []
                all_entropies = []

                for i in range(self._n):
                    obs_i    = obs_b[:, i, :]              # (batch, obs_dim)
                    mask_i   = masks_b[:, i, :]            # (batch, n_actions)
                    hidden_i = self.actor.init_hidden(batch_size).to(self._device)

                    dist, _ = self.actor(obs_i, hidden_i, mask_i)
                    lp      = dist.log_prob(actions_b[:, i])
                    ent     = dist.entropy()

                    all_log_probs.append(lp)
                    all_entropies.append(ent)

                # Mean over agents (parameter sharing)
                new_log_probs = torch.stack(all_log_probs, dim=1).mean(dim=1)
                old_log_probs = old_log_probs_b.mean(dim=1)
                entropy       = torch.stack(all_entropies, dim=1).mean()

                # -- PPO clip objective --
                ratio    = torch.exp(new_log_probs - old_log_probs)
                clipped  = torch.clamp(ratio, 1 - self._clip_eps, 1 + self._clip_eps)
                p_loss   = -torch.min(ratio * advantages_b, clipped * advantages_b).mean()

                # -- Value loss --
                joint_obs = obs_b.view(batch_size, -1)   # (batch, N×obs_dim)
                values    = self.critic(joint_obs).squeeze(-1)
                v_loss    = nn.functional.mse_loss(values, returns_b)

                # -- Total loss --
                loss = p_loss + 0.5 * v_loss - entropy_coef * entropy

                self._opt_actor.zero_grad()
                self._opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self._opt_actor.step()
                self._opt_critic.step()

                metrics["policy_loss"] += p_loss.item()
                metrics["value_loss"]  += v_loss.item()
                metrics["entropy"]     += entropy.item()
                metrics["n_updates"]   += 1

        # Average metrics
        n = max(metrics["n_updates"], 1)
        metrics["policy_loss"] /= n
        metrics["value_loss"]  /= n
        metrics["entropy"]     /= n

        self.buffer.reset()
        self._total_steps += UPDATE_EVERY
        return metrics

    def save(self, path: str) -> None:
        """Save actor and critic weights."""
        torch.save({
            "actor" : self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
        print(f"Saved weights → {path}")

    def load(self, path: str) -> None:
        """Load actor and critic weights."""
        ckpt = torch.load(path, map_location=self._device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        print(f"Loaded weights ← {path}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_joint_obs(self, obs: dict[int, np.ndarray]) -> torch.Tensor:
        """Concatenate all agents' obs for the centralised critic."""
        joint = np.concatenate([obs[i] for i in range(self._n)])
        return torch.FloatTensor(joint).unsqueeze(0).to(self._device)

    def _entropy_coef(self) -> float:
        """
        Linearly anneal entropy coefficient from ENT_START to ENT_END.
        Encourages exploration early, exploitation later.
        """
        progress = min(self._total_steps / ENT_ANNEAL_STEPS, 1.0)
        return ENT_START + (ENT_END - ENT_START) * progress