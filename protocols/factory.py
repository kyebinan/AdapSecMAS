# protocols/factory.py
# =============================================================================
# AdapSecMAS — ProtocolFactory
# Factory pattern with registry.
# OCP: register a new protocol without modifying existing code.
# DIP: callers depend on IProtocol, never on concrete classes.
# =============================================================================

from __future__ import annotations

import numpy as np

from core.interfaces import IProtocol
from protocols.freq_hop  import FreqHop
from protocols.ban_vote  import BanVote
from protocols.id_revoke import IdRevoke


class ProtocolFactory:
    """
    Creates protocol instances from a string key.

    Usage:
        hop    = ProtocolFactory.create("freq_hop")
        ban    = ProtocolFactory.create("ban_vote")
        revoke = ProtocolFactory.create("id_revoke")

    OCP: add a new protocol with register() — no if/elif chains.
    """

    _registry: dict[str, type[IProtocol]] = {
        "freq_hop" : FreqHop,
        "ban_vote" : BanVote,
        "id_revoke": IdRevoke,
    }

    @classmethod
    def create(cls, kind: str) -> IProtocol:
        klass = cls._registry.get(kind)
        if klass is None:
            known = ", ".join(cls._registry)
            raise ValueError(
                f"Unknown protocol: {kind!r}. Known: {known}"
            )
        return klass()

    @classmethod
    def register(cls, kind: str, klass: type[IProtocol]) -> None:
        """Register a new protocol type. OCP: extend without modifying."""
        cls._registry[kind] = klass

    @classmethod
    def create_default_set(cls) -> list[IProtocol]:
        """Create the standard set of 3 protocols for the training environment."""
        return [
            cls.create("freq_hop"),
            cls.create("ban_vote"),
            cls.create("id_revoke"),
        ]

    @classmethod
    def select(
        cls,
        protocols : list[IProtocol],
        obs       : np.ndarray,
    ) -> IProtocol | None:
        """
        Select the first protocol whose can_trigger() returns True.
        Called by NetworkEnv when action=6 is chosen.
        Priority: freq_hop > ban_vote > id_revoke.
        Returns None if no protocol is triggered.
        """
        for protocol in protocols:
            if protocol.can_trigger(obs):
                return protocol
        return None