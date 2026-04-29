# security/state_machine.py
# =============================================================================
# AdapSecMAS — SecurityLevelMachine (SLM)
# State pattern: agent behaviour changes based on the current security level.
#
# NOT part of the POSG game model — lives between observation and policy.
# Deterministic function of o_t^i — never learned by RL.
#
# SRP: only responsible for:
#   - computing threat_score from observation
#   - mapping score to level (with hysteresis on descent)
#   - returning action mask for the current level
# =============================================================================

from __future__ import annotations

import numpy as np

from security.levels import SecurityLevel, ACTION_MASKS
from core.constants import (
    SNR_THRESHOLD,
    SNR_MAX_NORM,
    FLOOD_RATE_THRESHOLD,
    FLOOD_RATE_ATTACK,
    THREAT_SCORE_ELEVATED,
    THREAT_SCORE_HIGH,
    THREAT_SCORE_CRITICAL,
    HYSTERESIS_STEPS,
)

# Normalised flood threshold used in obs space
_FLOOD_NORM_THRESHOLD: float = FLOOD_RATE_THRESHOLD / FLOOD_RATE_ATTACK
_SNR_NORM_THRESHOLD  : float = SNR_THRESHOLD / SNR_MAX_NORM


class SecurityLevelMachine:
    """
    Computes and manages the security level for one agent.

    State pattern:
      Each level defines available actions and action costs.
      Transitions are deterministic from threat_score(o_t^i).

    Hysteresis on descent:
      Level only drops after HYSTERESIS_STEPS consecutive steps
      below the lower threshold — prevents rapid oscillation
      when jammer is near the SNR boundary.

    Escalation is immediate — no hysteresis on the way up.
    """

    def __init__(self):
        self._level         : SecurityLevel = SecurityLevel.NORMAL
        self._steps_below   : int           = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def level(self) -> SecurityLevel:
        return self._level

    def update(self, obs: np.ndarray) -> SecurityLevel:
        """
        Compute threat score, update level with hysteresis, return new level.
        Call once per step with the agent's raw observation vector.

        obs layout (indices):
          0: snr_norm
          1: jammer_noise
          2: corrupt_rate
          3: flood_rate      (normalised)
          4: spoof_flag
          5: tx_norm
          6: level_norm      (previous level — ignored here)
        """
        score  = self._threat_score(obs)
        target = self._score_to_level(score)

        if target > self._level:
            # Escalate immediately
            self._level       = target
            self._steps_below = 0

        elif target < self._level:
            # Descend only after hysteresis window
            self._steps_below += 1
            if self._steps_below >= HYSTERESIS_STEPS:
                self._level       = SecurityLevel(self._level - 1)
                self._steps_below = 0

        else:
            self._steps_below = 0

        return self._level

    def action_mask(self) -> list[bool]:
        """Return the boolean action mask for the current level."""
        return ACTION_MASKS[self._level]

    def level_norm(self) -> float:
        """Return level normalised to [0, 1] for inclusion in obs vector."""
        return float(self._level) / 3.0

    def reset(self) -> None:
        """Reset to NORMAL — called at episode start."""
        self._level       = SecurityLevel.NORMAL
        self._steps_below = 0

    # ------------------------------------------------------------------
    # Private — threat score computation
    # ------------------------------------------------------------------

    @staticmethod
    def _threat_score(obs: np.ndarray) -> float:
        """
        Scalar threat score from normalised observation.

        Components:
          jam   = max(0, threshold - snr_norm) / threshold
          flood = max(0, flood_rate - threshold) / (1 - threshold)
          spoof = spoof_flag  ∈ {0, 1}

        Score range: [0, ~3]  (can exceed 1 when all attacks active)
        """
        snr_norm   = float(obs[0])
        flood_rate = float(obs[3])
        spoof_flag = float(obs[4])

        jam_component = max(0.0, _SNR_NORM_THRESHOLD - snr_norm) / max(_SNR_NORM_THRESHOLD, 1e-9)
        fld_component = max(0.0, flood_rate - _FLOOD_NORM_THRESHOLD) / max(1.0 - _FLOOD_NORM_THRESHOLD, 1e-9)
        spf_component = spoof_flag

        return jam_component + fld_component + spf_component

    @staticmethod
    def _score_to_level(score: float) -> SecurityLevel:
        """Map threat score to a discrete security level."""
        if score >= THREAT_SCORE_CRITICAL:
            return SecurityLevel.CRITICAL
        if score >= THREAT_SCORE_HIGH:
            return SecurityLevel.HIGH
        if score >= THREAT_SCORE_ELEVATED:
            return SecurityLevel.ELEVATED
        return SecurityLevel.NORMAL