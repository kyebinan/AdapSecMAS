# protocols/id_revoke.py
# =============================================================================
# AdapSecMAS — IdRevoke protocol
# Anti-spoofing: distributed identity revocation without central PKI.
# Inspired by Baudet (2023) MAKI architecture.
#
# Steps:
#   1. Agent i detects spoof_flag=1 (seq anomaly) → triggers ID-REVOKE
#   2. Identifies the forged sender_id from validation pipeline
#   3. Broadcasts REVOKE_REQ with proof (seq delta)
#   4. Neighbours verify proof locally
#   5. Quorum ceil(2n/3) → forged_id added to distributed CRL
#   6. CRL propagated via gossip in network_state
#
# Stronger quorum than BAN-VOTE (2/3 vs 1/2) — revocation is hard to undo.
# No central PKI — each agent maintains a local CRL copy.
# SRP: only handles spoofing mitigation.
# =============================================================================

from __future__ import annotations

import math

import numpy as np

from protocols.base import BaseProtocol
from core.interfaces import ProtocolResult
from core.constants import REVOKE_QUORUM


class IdRevoke(BaseProtocol):
    """
    Distributed identity revocation — anti-spoofing.

    Triggered when spoof_flag = 1 (seq counter anomaly detected).
    Requires ceil(2n/3) confirmations from neighbours to revoke.

    Effect on network_state:
        network_state["crl"].add(forged_id)        # Certificate Revocation List
        network_state["revoked_steps"][forged_id] = REVOKE_DURATION
    """

    _REVOKE_DURATION: int = 100   # steps before revocation expires

    @property
    def name(self) -> str:
        return "ID-REVOKE"

    def can_trigger(self, obs: np.ndarray) -> bool:
        """Trigger if a seq anomaly (spoof) has been detected."""
        return self._spoof_flag(obs) > 0.5

    def _run(
        self,
        agent_id     : int,
        neighbours   : list[int],
        network_state: dict,
    ) -> ProtocolResult:
        """
        Identify the forged ID, collect confirmations, update CRL if quorum reached.
        """
        forged_id = self._identify_forged_id(agent_id, network_state)
        if forged_id is None:
            return ProtocolResult(
                success=False,
                reason="no forged identity identified",
            )

        # Do not revoke a legitimate agent
        legitimate_ids = network_state.get("legitimate_ids", set())
        if forged_id in legitimate_ids:
            return ProtocolResult(
                success=False,
                target_id=forged_id,
                reason=f"agent {forged_id} is in the legitimate set — revocation blocked",
            )

        confirmations = self._collect_confirmations(
            forged_id, neighbours, network_state
        )
        quorum = math.ceil(len(neighbours) * REVOKE_QUORUM)
        quorum = max(quorum, 1)

        if confirmations < quorum:
            return ProtocolResult(
                success=False,
                reason=f"quorum not reached ({confirmations}/{quorum})",
            )

        # Add to distributed CRL
        crl            = network_state.setdefault("crl", set())
        revoked_steps  = network_state.setdefault("revoked_steps", {})
        crl.add(forged_id)
        revoked_steps[forged_id] = self._REVOKE_DURATION

        return ProtocolResult(
            success   = True,
            target_id = forged_id,
            reason    = (
                f"identity {forged_id} revoked "
                f"({confirmations}/{len(neighbours)} confirmations)"
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_forged_id(
        agent_id     : int,
        network_state: dict,
    ) -> int | None:
        """
        Find the sender_id flagged as forged in the agent's validation log.
        Returns None if no forged ID is recorded.
        """
        spoof_log = network_state.get("spoof_log", {})
        # spoof_log[agent_id] = list of (forged_id, seq_delta) tuples
        agent_log = spoof_log.get(agent_id, [])
        if not agent_log:
            return None

        # Most recently detected forged ID
        forged_id, _seq_delta = agent_log[-1]
        return forged_id

    @staticmethod
    def _collect_confirmations(
        forged_id    : int,
        neighbours   : list[int],
        network_state: dict,
    ) -> int:
        """
        Count how many neighbours have also flagged forged_id as a spoofer.
        Each neighbour checks its own spoof_log independently.
        """
        spoof_log     = network_state.get("spoof_log", {})
        confirmations = 1   # agent i confirms itself

        for nb in neighbours:
            nb_log    = spoof_log.get(nb, [])
            nb_forged = {fid for fid, _ in nb_log}
            if forged_id in nb_forged:
                confirmations += 1

        return confirmations