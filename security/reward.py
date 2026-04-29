# security/reward.py
# =============================================================================
# AdapSecMAS — RewardComputer
# SRP: only responsible for computing R_team from StepMetrics.
# Never touches environment state directly.
# Never modifies agent state.
#
# R_team = M(t) - C_sec(t) - C_act(t)
#
# All weights come from constants.py — no magic numbers here.
# =============================================================================

from __future__ import annotations

from core.metrics import StepMetrics
from security.levels import SecurityLevel
from core.constants import (
    ALPHA_DELIVERY_RATE,
    ALPHA_LINKS_HEALTHY,
    ALPHA_PROTOCOL_OK,
    BETA_JAM_LOSS,
    BETA_QUEUE_OVER,
    BETA_SPOOF_ACCEPT,
    BETA_MISMATCH,
    BETA_PROTO_FAIL,
    ACTION_COST_TABLE,
)


class RewardComputer:
    """
    Computes the cooperative team reward R_team for one step.

    R_team = M - C_sec - C_act

    M      : mission success signal  (delta-based — rewards improvement)
    C_sec  : security exposure       (penalises actual attack damage)
    C_act  : action cost             (level-dependent — discourages over-reaction)

    SRP: receives a StepMetrics snapshot and the chosen actions.
    Does not read from the environment or modify any state.
    """

    def compute(
        self,
        metrics : StepMetrics,
        actions : dict[int, int],
        levels  : dict[int, SecurityLevel],
    ) -> float:
        """
        Compute R_team for one simulation step.

        Parameters
        ----------
        metrics : step snapshot produced by NetworkEnv
        actions : {agent_id: action_id} chosen this step
        levels  : {agent_id: SecurityLevel} current level per agent

        Returns
        -------
        float — team reward (positive is good)
        """
        m     = self._mission(metrics)
        c_sec = self._security_exposure(metrics)
        c_act = self._action_cost(actions, levels)
        return m - c_sec - c_act

    # ------------------------------------------------------------------
    # M — mission success
    # ------------------------------------------------------------------

    @staticmethod
    def _mission(metrics: StepMetrics) -> float:
        """
        Delta-based: rewards improvement over the previous step.
        Dense signal: n_links_healthy gives a signal at every step.
        """
        return (
            ALPHA_DELIVERY_RATE * metrics.delta_delivery_rate
          + ALPHA_LINKS_HEALTHY * metrics.n_links_healthy
          + ALPHA_PROTOCOL_OK   * metrics.delta_protocol_success
        )

    # ------------------------------------------------------------------
    # C_sec — security exposure
    # ------------------------------------------------------------------

    @staticmethod
    def _security_exposure(metrics: StepMetrics) -> float:
        """
        Penalises actual damage caused by each attack family.
        level_mismatch is the primary escalation signal:
          if threat > level → agent is under-reacting → pay BETA_MISMATCH per gap unit.
        """
        return (
            BETA_JAM_LOSS    * metrics.n_msgs_lost_to_jam
          + BETA_QUEUE_OVER  * metrics.n_queue_overflows
          + BETA_SPOOF_ACCEPT* metrics.n_spoof_accepted
          + BETA_MISMATCH    * metrics.level_mismatch
          + BETA_PROTO_FAIL  * metrics.n_protocol_failed
        )

    # ------------------------------------------------------------------
    # C_act — action cost (level-dependent)
    # ------------------------------------------------------------------

    @staticmethod
    def _action_cost(
        actions: dict[int, int],
        levels : dict[int, SecurityLevel],
    ) -> float:
        """
        Sum of action costs across all agents.
        Cost is lower when the level justifies the action.
        At CRITICAL, noop costs 0.20 — inaction is penalised.
        """
        total = 0.0
        for agent_id, action in actions.items():
            level      = int(levels.get(agent_id, SecurityLevel.NORMAL))
            total     += ACTION_COST_TABLE[level][action]
        return total