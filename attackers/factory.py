# attackers/factory.py
# =============================================================================
# AdapSecMAS — AttackerFactory
# Factory pattern with registry.
# OCP: register a new attacker type without modifying existing code.
# DIP: callers depend on IAttacker, never on concrete classes.
# =============================================================================

from __future__ import annotations

from core.interfaces import IAttacker
from core.constants import ARENA_WIDTH, ARENA_HEIGHT
from attackers.jammer  import JammerAgent
from attackers.flooder import FloodAgent
from attackers.spoofer import SpoofAgent


class AttackerFactory:
    """
    Creates attacker instances from a string key and keyword arguments.

    Usage:
        jammer  = AttackerFactory.create("jammer")
        flooder = AttackerFactory.create("flooder", agent_id=19, victim_id=5)
        spoofer = AttackerFactory.create("spoofer", victim_id=3)

    OCP: add a new attacker type with register() — no if/elif chains.
    """

    _registry: dict[str, type[IAttacker]] = {
        "jammer" : JammerAgent,
        "flooder": FloodAgent,
        "spoofer": SpoofAgent,
    }

    @classmethod
    def create(cls, kind: str, **kwargs) -> IAttacker:
        """
        Instantiate an attacker of the given kind.
        kwargs are passed directly to the concrete constructor.
        """
        klass = cls._registry.get(kind)
        if klass is None:
            known = ", ".join(cls._registry)
            raise ValueError(
                f"Unknown attacker type: {kind!r}. Known types: {known}"
            )
        return klass(**kwargs)

    @classmethod
    def register(cls, kind: str, klass: type[IAttacker]) -> None:
        """
        Register a new attacker type.
        OCP: extend without modifying this class.

        Example:
            AttackerFactory.register("drone_jammer", DroneJammerAgent)
        """
        cls._registry[kind] = klass

    @classmethod
    def create_default_set(cls, rng_seed: int | None = None) -> list[IAttacker]:
        """
        Convenience: create the standard set of 3 attackers
        used in the MARL training environment.
        """
        return [
            cls.create(
                "jammer",
                initial_pos=(ARENA_WIDTH / 2, ARENA_HEIGHT / 2),
                rng_seed=rng_seed,
            ),
            cls.create(
                "flooder",
                agent_id=20,   # ID outside the 0-19 protagonist range
                initial_pos=(ARENA_WIDTH / 4, ARENA_HEIGHT / 4),
                rng_seed=rng_seed,
            ),
            cls.create(
                "spoofer",
                victim_id=0,   # spoofs agent 0 by default
                initial_pos=(3 * ARENA_WIDTH / 4, ARENA_HEIGHT / 4),
                rng_seed=rng_seed,
            ),
        ]