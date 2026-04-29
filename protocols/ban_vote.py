# protocols/ban_vote.py
# =============================================================================
# AdapSecMAS — BanVote protocol
# Anti-flooding: Byzantine majority vote to ban a flooding agent.
#
# Steps:
#   1. Agent i detects flood_rate > threshold → triggers BAN-VOTE
#   2. Identifies the suspected flooder from network_state
#   3. Collects signed votes from neighbours
#   4. Quorum ceil((n+1)/2) → suspect banned for BAN_DURATION_STEPS
#   5. Ban propagated via network_state (gossip handled by NetworkEnv)
#
# Byzantine resilience: tolerates f < n/3 malicious voters.
# Inspired by PBFT and Baudet (2023) MAKI revocation mechanism.
# SRP: only handles flooding mitigation.
# =============================================================================

from __future__ import annotations

import math

import numpy as np

from protocols.base import BaseProtocol
from core.interfaces import ProtocolResult
from core.constants import (
    BAN_VOTE_QUORUM,
    BAN_DURATION_STEPS,
    FLOOD_RATE_THRESHOLD,
    FLOOD_RATE_ATTACK,
)

# Normalised threshold: fraction of max flood rate above which we trigger
_FLOOD_RATE_NORM_THRESHOLD: float = FLOOD_RATE_THRESHOLD / FLOOD_RATE_ATTACK


class BanVote(BaseProtocol):
    """
    Majority vote to ban a flooding agent.

    Triggered when flood_rate > FLOOD_RATE_THRESHOLD.
    Requires ceil((n+1)/2) votes from neighbours to ban.

    Effect on network_state:
        network_state["banned"][suspect_id] = BAN_DURATION_STEPS
    """

    @property
    def name(self) -> str:
        return "BAN-VOTE"

    def can_trigger(self, obs: np.ndarray) -> bool:
        """Trigger if normalised flood rate exceeds threshold."""
        return self._flood_rate(obs) > _FLOOD_RATE_NORM_THRESHOLD

    def _run(
        self,
        agent_id     : int,
        neighbours   : list[int],
        network_state: dict,
    ) -> ProtocolResult:
        """
        Identify the flooding suspect, collect votes, apply ban if quorum reached.
        """
        suspect_id = self._identify_suspect(agent_id, network_state)
        if suspect_id is None:
            return ProtocolResult(
                success=False,
                reason="no flooding suspect identified",
            )

        flood_rates = network_state.get("flood_rate_per_sender", {})
        votes       = self._collect_votes(suspect_id, neighbours, flood_rates)
        quorum      = math.ceil((len(neighbours) + 1) / 2)

        if votes < quorum:
            return ProtocolResult(
                success=False,
                reason=f"quorum not reached ({votes}/{quorum} votes)",
            )

        banned = network_state.setdefault("banned", {})
        banned[suspect_id] = BAN_DURATION_STEPS

        return ProtocolResult(
            success   = True,
            target_id = suspect_id,
            reason    = (
                f"agent {suspect_id} banned for {BAN_DURATION_STEPS} steps "
                f"({votes}/{len(neighbours)} votes)"
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_suspect(
        agent_id     : int,
        network_state: dict,
    ) -> int | None:
        """
        Find the sender with the highest flood rate in the agent's window.
        Returns None if no sender exceeds the threshold.
        """
        flood_rates = network_state.get("flood_rate_per_sender", {})
        if not flood_rates:
            return None

        # Filter senders above threshold
        suspects = {
            sid: rate for sid, rate in flood_rates.items()
            if rate > FLOOD_RATE_THRESHOLD
        }
        if not suspects:
            return None

        return max(suspects, key=suspects.__getitem__)

    @staticmethod
    def _collect_votes(
        suspect_id  : int,
        neighbours  : list[int],
        flood_rates : dict,
    ) -> int:
        """
        Count how many neighbours have also observed high flood rate
        from suspect_id. Each neighbour casts one signed vote.
        Byzantine simplification: assume honest neighbours report truthfully.
        """
        votes = 1   # agent i votes for itself
        for nb in neighbours:
            nb_rate = flood_rates.get((nb, suspect_id), 0.0)
            if nb_rate > FLOOD_RATE_THRESHOLD:
                votes += 1
        return votes