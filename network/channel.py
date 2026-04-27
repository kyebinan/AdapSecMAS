# network/channel.py

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from core.interfaces import IChannel, IAttacker
from core.message import Message
from core.constants import (
    SNR_THRESHOLD,
    NEAR_THRESHOLD_WINDOW,
    NEAR_THRESHOLD_MAX_FAIL,
    PATH_LOSS_EXP,
    NOISE_FLOOR,
    TX_POWER_DEFAULT,
)

if TYPE_CHECKING:
    pass


class WirelessChannel(IChannel):
    """
    Wireless channel model for IEEE 802.11 / 2.4 GHz simulation.

    Models:
      - Path loss with exponent α (urban environment)
      - Ambient noise floor
      - Jammer interference field J(x,y,t) — spatial, continuous
      - Probabilistic packet loss near the SNR threshold
      - Attacker message injection (spoofing, flooding)

    """

    def __init__(
        self,
        path_loss_exp      : float = PATH_LOSS_EXP,
        noise_floor        : float = NOISE_FLOOR,
        snr_threshold      : float = SNR_THRESHOLD,
        near_window        : float = NEAR_THRESHOLD_WINDOW,
        near_max_fail_prob : float = NEAR_THRESHOLD_MAX_FAIL,
        rng_seed           : int | None = None,
    ):
        self._alpha         = path_loss_exp
        self._noise_floor   = noise_floor
        self._threshold     = snr_threshold
        self._near_window   = near_window
        self._near_max_fail = near_max_fail_prob
        self._attackers     : list[IAttacker] = []
        self._rng           = random.Random(rng_seed)

    # ------------------------------------------------------------------
    # IChannel interface
    # ------------------------------------------------------------------

    def snr(
        self,
        tx_pos  : tuple[float, float],
        rx_pos  : tuple[float, float],
        tx_power: float = TX_POWER_DEFAULT,
    ) -> float:
        """
        Compute SNR at rx_pos for a transmission from tx_pos.

        SNR = tx_power / (d^alpha x (noise_floor + J(rx_pos)))
        """
        dist  = self._distance(tx_pos, rx_pos)
        noise = self._noise_floor + self.jammer_noise_at(rx_pos)
        return self._snr_formula(tx_power, dist, self._alpha, noise)

    def deliver(
        self,
        message : Message,
        tx_pos  : tuple[float, float],
        rx_pos  : tuple[float, float],
        tx_power: float = TX_POWER_DEFAULT,
    ) -> tuple[Message | None, float]:
        """
        Attempt to deliver a message through the channel.

        Steps:
          1. Compute SNR at receiver position
          2. Let attackers inject/corrupt the message
          3. Decide success based on SNR

        Returns
        -------
        (message, snr)  if delivered (message may be altered by attacker)
        (None,    snr)  if dropped
        """
        snr_val = self.snr(tx_pos, rx_pos, tx_power)
        altered = self._apply_attacker_injection(message)
        success = self._decide_success(snr_val)

        if success:
            return altered, snr_val
        return None, snr_val

    def jammer_noise_at(self, pos: tuple[float, float]) -> float:
        """
        Total jammer interference power at position pos.
        J(x,y,t) = Sum jammer_i.noise_at(pos) for all active jammers.
        """
        return sum(a.noise_at(pos) for a in self._attackers)

    def step(self, dt: float) -> None:
        """
        Advance channel state.
        Attackers are stepped by NetworkEnv — channel itself is stateless.
        """
        pass

    # ------------------------------------------------------------------
    # Attacker management
    # ------------------------------------------------------------------

    def add_attacker(self, attacker: IAttacker) -> None:
        """
        Register an attacker with the channel.
        DIP: depends on IAttacker abstraction, not concrete types.
        """
        self._attackers.append(attacker)

    def remove_attacker(self, attacker: IAttacker) -> None:
        self._attackers.remove(attacker)

    # ------------------------------------------------------------------
    # Peer message noise — used by GossipMediator
    # ------------------------------------------------------------------

    def apply_comm_noise(
        self,
        value  : float,
        rx_pos : tuple[float, float],
        snr_val: float,
    ) -> float | None:
        """
        Degrade a peer message scalar with jammer noise.
        Returns None if the message is lost (SNR too low).
        Used by GossipMediator to degrade m_i_t before delivery to j.
        """
        if snr_val < self._threshold - self._near_window:
            return None

        noise_ratio = self.jammer_noise_at(rx_pos) / max(self._noise_floor, 1e-9)
        sigma       = 0.05 * noise_ratio
        return value + self._rng.gauss(0, sigma)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snr_formula(
        tx_power: float,
        distance: float,
        alpha   : float,
        noise   : float,
    ) -> float:
        """SNR = P_tx / (d^alpha x noise). Clamps to avoid division by zero."""
        d_clamped = max(distance, 1e-6)
        return tx_power / (d_clamped ** alpha * max(noise, 1e-9))

    @staticmethod
    def _distance(
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        """Euclidean distance between two 2D positions."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _decide_success(self, snr_val: float) -> bool:
        """
        Three zones:
          SNR >= threshold                    → always delivered
          threshold-window <= SNR < threshold → probabilistic
          SNR < threshold - window            → always dropped
        """
        if snr_val >= self._threshold:
            return True

        lower = self._threshold - self._near_window
        if snr_val < lower:
            return False

        margin = snr_val - lower
        p_fail = self._near_max_fail * (1.0 - margin / self._near_window)
        return self._rng.random() >= p_fail

    def _apply_attacker_injection(self, message: Message) -> Message:
        """
        Let each attacker optionally alter the message in transit.
        Jammer returns the original. Spoofer may return a forged copy.
        """
        current = message
        for attacker in self._attackers:
            result = attacker.inject(current)
            if result is not None:
                current = result
        return current