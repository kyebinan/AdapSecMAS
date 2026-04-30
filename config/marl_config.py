# config/marl_config.py
# =============================================================================
# AdapSecMAS — MARL / MAPPO configuration
# All learning hyperparameters in one place.
# frozen=True: prevents accidental mutation during training.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from core.constants import (
    GAMMA,
    LAMBDA_GAE,
    CLIP_EPS,
    LR_ACTOR,
    LR_CRITIC,
    UPDATE_EVERY,
    PPO_EPOCHS,
    BATCH_SIZE,
    HIDDEN_SIZE,
    ENT_START,
    ENT_END,
    ENT_ANNEAL_STEPS,
    DIM_OBS_TOTAL,
    N_ACTIONS,
)


@dataclass(frozen=True)
class MARLConfig:
    """
    MAPPO hyperparameters.
    Passed to MAPPOTrainer at construction.

    Curriculum phases are defined in train.py — not here,
    because they depend on episode count which is a training concern.
    """
    # Network architecture
    obs_dim     : int   = DIM_OBS_TOTAL
    action_dim  : int   = N_ACTIONS
    hidden_size : int   = HIDDEN_SIZE

    # PPO
    gamma       : float = GAMMA
    lambda_gae  : float = LAMBDA_GAE
    clip_eps    : float = CLIP_EPS
    ppo_epochs  : int   = PPO_EPOCHS
    batch_size  : int   = BATCH_SIZE
    update_every: int   = UPDATE_EVERY

    # Optimisers
    lr_actor    : float = LR_ACTOR
    lr_critic   : float = LR_CRITIC

    # Entropy annealing
    entropy_start : float = ENT_START
    entropy_end   : float = ENT_END
    entropy_anneal: int   = ENT_ANNEAL_STEPS

    # Device
    device: str = "auto"   # "auto" | "cpu" | "cuda"

    def summary(self) -> str:
        return (
            f"MARLConfig("
            f"gamma={self.gamma}, "
            f"clip={self.clip_eps}, "
            f"lr_actor={self.lr_actor}, "
            f"lr_critic={self.lr_critic}, "
            f"hidden={self.hidden_size}, "
            f"update_every={self.update_every}"
            f")"
        )