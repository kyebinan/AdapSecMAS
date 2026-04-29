# protocols/freq_hop.py
# =============================================================================
# AdapSecMAS — FreqHop protocol
# Anti-jamming: coordinated channel switch via local majority vote.
#
# Steps:
#   1. Agent i detects low SNR → triggers FREQ-HOP (action=6)
#   2. Queries neighbours: how many also have degraded SNR?
#   3. If local majority confirms → all switch to channel+1
#   4. tx_power boosted during transition window
#
# No central coordinator — k-local majority rule.
# SRP: only handles jamming mitigation.
# =============================================================================

from __future__ import annotations

import numpy as np

from protocols.base import BaseProtocol
from core.interfaces import ProtocolResult
from core.constants import (
    SNR_THRESHOLD,
    SNR_MAX_NORM,
    FREQ_HOP_QUORUM,
    TX_POWER_DEFAULT,
)

# Number of non-overlapping 2.4 GHz channels available
_N_CHANNELS: int = 3   # channels 1, 6, 11


class FreqHop(BaseProtocol):
    """
    Frequency hopping protocol — anti-jamming.

    Triggered when snr_norm < threshold (jammer detected locally).
    Requires FREQ_HOP_QUORUM fraction of neighbours to confirm
    degraded SNR before switching channels.

    Effect on network_state:
        network_state["channel"][agent_id] = (channel + 1) % N_CHANNELS
        network_state["tx_power"][agent_id] = TX_POWER_DEFAULT * 8  (boost during switch)
    """

    @property
    def name(self) -> str:
        return "FREQ-HOP"

    def can_trigger(self, obs: np.ndarray) -> bool:
        """Trigger if SNR is below the normalised threshold."""
        snr_norm_threshold = SNR_THRESHOLD / SNR_MAX_NORM
        return self._snr_norm(obs) < snr_norm_threshold

    def _run(
        self,
        agent_id     : int,
        neighbours   : list[int],
        network_state: dict,
    ) -> ProtocolResult:
        """
        Count neighbours with degraded SNR.
        If quorum reached → switch channel for all confirming agents.
        """
        snr_per_agent  = network_state.get("snr", {})
        channels       = network_state.setdefault("channel", {})
        tx_powers      = network_state.setdefault("tx_power", {})

        snr_threshold  = SNR_THRESHOLD
        confirming     = [
            nb for nb in neighbours
            if snr_per_agent.get(nb, snr_threshold + 1) < snr_threshold
        ]

        quorum_needed = max(1, int(len(neighbours) * FREQ_HOP_QUORUM))

        if len(confirming) < quorum_needed:
            return ProtocolResult(
                success=False,
                reason=f"quorum not reached ({len(confirming)}/{quorum_needed})",
            )

        # Switch channel for agent i and all confirming neighbours
        current_channel = channels.get(agent_id, 0)
        new_channel     = (current_channel + 1) % _N_CHANNELS

        participants = [agent_id] + confirming
        for pid in participants:
            channels[pid]  = new_channel
            tx_powers[pid] = TX_POWER_DEFAULT * 8   # boost during transition

        return ProtocolResult(
            success   = True,
            target_id = None,
            reason    = (
                f"hopped to channel {new_channel} "
                f"({len(participants)} agents)"
            ),
        )