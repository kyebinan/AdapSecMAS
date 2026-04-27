# network/gossip.py

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from core.message import PeerMessage
from core.constants import DIM_COMM

if TYPE_CHECKING:
    from network.channel import WirelessChannel


class GossipMediator:
    """
    Mediates peer-to-peer communication between agents.

    At each step:
      1. Each agent broadcasts its PeerMessage via broadcast()
      2. Mediator routes and degrades each message through the channel
      3. Each agent collects its received messages via collect()
      4. NetworkEnv aggregates them into agg_comm for the observation vector

    Agents are fully decoupled - they never address each other directly.
    """

    def __init__(self, n_agents: int):
        self._n     = n_agents
        self._inbox : dict[int, list[PeerMessage]] = {
            i: [] for i in range(n_agents)
        }

    def broadcast(
        self,
        sender_id : int,
        message   : PeerMessage,
        positions : dict[int, tuple[float, float]],
        tx_powers : dict[int, float],
        channel   : "WirelessChannel",
    ) -> None:
        """
        Agent sender_id broadcasts m_i_t to all other agents.
        Each scalar is degraded by channel noise before delivery.
        """
        tx_pos   = positions[sender_id]
        tx_power = tx_powers[sender_id]

        for receiver_id in range(self._n):
            if receiver_id == sender_id:
                continue

            rx_pos  = positions[receiver_id]
            snr_val = channel.snr(tx_pos, rx_pos, tx_power)
            noisy   = self._degrade_message(message, rx_pos, snr_val, channel)

            if noisy is not None:
                self._inbox[receiver_id].append(noisy)

    def collect(self, agent_id: int) -> list[PeerMessage]:
        """Collect all messages received by agent_id this step. Clears inbox."""
        messages = self._inbox[agent_id]
        self._inbox[agent_id] = []
        return messages

    def aggregate(self, messages: list[PeerMessage]) -> np.ndarray:
        """
        Aggregate received PeerMessages into a single vector (element-wise mean).
        Returns zero vector if no messages received.
        Output shape: (DIM_COMM,) = (5,)
        """
        if not messages:
            return np.zeros(DIM_COMM, dtype=np.float32)

        arrays = [np.array(m.to_array(), dtype=np.float32) for m in messages]
        return np.mean(arrays, axis=0)

    def reset(self) -> None:
        """Clear all inboxes - called at episode start."""
        for i in range(self._n):
            self._inbox[i] = []

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _degrade_message(
        self,
        message : PeerMessage,
        rx_pos  : tuple[float, float],
        snr_val : float,
        channel : "WirelessChannel",
    ) -> PeerMessage | None:
        """
        Apply channel noise to each scalar in the message.
        Returns None if the message is lost (SNR too low).
        """
        scalars  = message.to_array()
        degraded = []

        for value in scalars:
            noisy = channel.apply_comm_noise(value, rx_pos, snr_val)
            if noisy is None:
                return None
            degraded.append(float(np.clip(noisy, 0.0, 1.0)))

        return PeerMessage(
            sender_id    = message.sender_id,
            snr_norm     = degraded[0],
            corrupt_rate = degraded[1],
            flood_rate   = degraded[2],
            spoof_flag   = degraded[3],
            level_norm   = degraded[4],
        )