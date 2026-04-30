# production/delivery_env.py
# =============================================================================
# AdapSecMAS — DeliveryEnv
# Phase 2 production environment.
# Wraps NetworkEnv with a functional delivery layer:
#   - Trucks follow A* routes between depot and drop points
#   - Drones fly waypoint plans to rooftop pads
#   - Dispatcher assigns packages using EDF + proximity
#
# The trained MARL policy runs on top — agents defend the network
# while the delivery task runs in parallel.
#
# SRP: orchestrates delivery logic only.
#      Network security logic stays in NetworkEnv.
# =============================================================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto

from simulation.network_env import NetworkEnv
from agents.mappo           import MAPPOTrainer
from core.constants         import N_AGENTS, ARENA_WIDTH, ARENA_HEIGHT


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class AgentType(Enum):
    TRUCK = auto()
    DRONE = auto()


class TaskStatus(Enum):
    PENDING   = auto()
    ASSIGNED  = auto()
    IN_TRANSIT= auto()
    DELIVERED = auto()
    FAILED    = auto()


@dataclass
class Package:
    pkg_id      : int
    origin      : tuple[float, float]
    destination : tuple[float, float]
    deadline    : float          # sim time by which delivery must complete
    requires_drone: bool         # True = rooftop pad, False = ground drop
    status      : TaskStatus = TaskStatus.PENDING
    assigned_to : int | None = None


@dataclass
class DeliveryAgent:
    agent_id   : int
    agent_type : AgentType
    pos        : tuple[float, float]
    battery    : float = 1.0         # [0, 1]
    carrying   : Package | None = None
    waypoints  : list[tuple[float, float]] = field(default_factory=list)
    speed      : float = 60.0        # px/s


# ---------------------------------------------------------------------------
# Dispatcher — EDF + proximity assignment
# ---------------------------------------------------------------------------

class Dispatcher:
    """
    Assigns packages to agents using Earliest Deadline First + proximity.
    SRP: only responsible for task assignment — no movement, no network.
    """

    def assign(
        self,
        packages: list[Package],
        agents  : list[DeliveryAgent],
        sim_time: float,
    ) -> None:
        """Assign pending packages to idle agents."""
        idle     = [a for a in agents if a.carrying is None and a.battery > 0.2]
        pending  = sorted(
            [p for p in packages if p.status == TaskStatus.PENDING],
            key=lambda p: p.deadline,   # EDF
        )

        for pkg in pending:
            candidates = [
                a for a in idle
                if self._type_matches(a, pkg)
            ]
            if not candidates:
                continue

            # Assign to closest idle agent
            best = min(candidates, key=lambda a: self._dist(a.pos, pkg.origin))
            pkg.assigned_to = best.agent_id
            pkg.status      = TaskStatus.ASSIGNED
            best.carrying   = pkg
            best.waypoints  = [pkg.origin, pkg.destination]
            idle.remove(best)

    @staticmethod
    def _type_matches(agent: DeliveryAgent, pkg: Package) -> bool:
        if pkg.requires_drone:
            return agent.agent_type == AgentType.DRONE
        return agent.agent_type == AgentType.TRUCK

    @staticmethod
    def _dist(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# DeliveryEnv — Facade
# ---------------------------------------------------------------------------

class DeliveryEnv:
    """
    Production environment — wraps NetworkEnv with delivery logic.

    Each step:
      1. MARL policy acts on network observations (security layer)
      2. Delivery agents move toward their waypoints (functional layer)
      3. Dispatcher assigns new packages to idle agents
      4. Network step executes with MARL actions

    Provides combined state for the visualiser.
    """

    N_TRUCKS = 5
    N_DRONES = 5

    def __init__(
        self,
        weights_path: str,
        seed        : int = 0,
        dt          : float = 1 / 60,   # real-time rendering
    ):
        self._dt      = dt
        self._rng     = random.Random(seed)
        self._time    = 0.0

        # Network security layer
        self._net_env  = NetworkEnv(n_agents=N_AGENTS, rng_seed=seed)
        self._trainer  = MAPPOTrainer(device="auto")
        self._trainer.load(weights_path)
        self._trainer.actor.eval()

        # Delivery agents
        self._agents = self._spawn_agents()

        # Dispatcher
        self._dispatcher = Dispatcher()

        # Package queue
        self._packages   : list[Package] = []
        self._pkg_counter: int = 0

        # Depot position (top-left area)
        self._depot_pos = (80.0, 80.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset both layers and return initial state."""
        self._time     = 0.0
        obs            = self._net_env.reset()
        self._trainer.reset_hiddens()
        self._packages = []
        self._agents   = self._spawn_agents()
        self._spawn_packages(n=10)
        return self._state(obs)

    def step(self) -> dict:
        """Advance one frame."""
        self._time += self._dt

        # 1. MARL policy — security actions
        obs = self._net_env._build_observations()
        actions, _, _ = self._trainer.act(obs)

        # 2. Network step
        new_obs, reward, _, info = self._net_env.step(actions)

        # 3. Move delivery agents
        self._move_agents()

        # 4. Check deliveries
        self._check_deliveries()

        # 5. Assign new packages
        self._dispatcher.assign(self._packages, self._agents, self._time)

        # 6. Spawn new packages occasionally
        if self._rng.random() < 0.02:   # ~2% chance per frame
            self._spawn_packages(n=1)

        return self._state(new_obs, info=info, reward=reward)

    # ------------------------------------------------------------------
    # State for visualiser
    # ------------------------------------------------------------------

    def _state(
        self,
        obs    : dict,
        info   : dict | None = None,
        reward : float = 0.0,
    ) -> dict:
        """Build combined state dict for map_view and network_view."""
        return {
            # Delivery layer
            "agents"        : self._agents,
            "packages"      : self._packages,
            "depot_pos"     : self._depot_pos,
            "sim_time"      : self._time,

            # Network layer
            "positions"     : self._net_env.positions,
            "network_state" : self._net_env.network_state,
            "jammer_pos"    : self._net_env._jammer.pos,
            "jammer_radius" : self._net_env._jammer._radius,
            "obs"           : obs,
            "reward"        : reward,
            "info"          : info or {},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_agents(self) -> list[DeliveryAgent]:
        agents = []
        for i in range(self.N_TRUCKS):
            agents.append(DeliveryAgent(
                agent_id   = i,
                agent_type = AgentType.TRUCK,
                pos        = (
                    self._rng.uniform(60, 300),
                    self._rng.uniform(60, 300),
                ),
                speed = 50.0,
            ))
        for i in range(self.N_DRONES):
            agents.append(DeliveryAgent(
                agent_id   = self.N_TRUCKS + i,
                agent_type = AgentType.DRONE,
                pos        = (
                    self._rng.uniform(60, 300),
                    self._rng.uniform(60, 300),
                ),
                speed = 80.0,
            ))
        return agents

    def _spawn_packages(self, n: int) -> None:
        for _ in range(n):
            requires_drone = self._rng.random() < 0.4
            self._packages.append(Package(
                pkg_id        = self._pkg_counter,
                origin        = self._depot_pos,
                destination   = (
                    self._rng.uniform(100, ARENA_WIDTH  - 100),
                    self._rng.uniform(100, ARENA_HEIGHT - 100),
                ),
                deadline      = self._time + self._rng.uniform(60, 300),
                requires_drone= requires_drone,
            ))
            self._pkg_counter += 1

    def _move_agents(self) -> None:
        """Move each agent toward its next waypoint."""
        for agent in self._agents:
            if not agent.waypoints:
                continue

            target    = agent.waypoints[0]
            dx        = target[0] - agent.pos[0]
            dy        = target[1] - agent.pos[1]
            dist      = math.sqrt(dx * dx + dy * dy)
            step_dist = agent.speed * self._dt

            if dist <= step_dist:
                agent.pos = target
                agent.waypoints.pop(0)
            else:
                agent.pos = (
                    agent.pos[0] + dx / dist * step_dist,
                    agent.pos[1] + dy / dist * step_dist,
                )

            # Battery drain
            agent.battery = max(0.0, agent.battery - 0.0001 * self._dt)

    def _check_deliveries(self) -> None:
        """Mark packages as delivered when agent reaches destination."""
        for agent in self._agents:
            if agent.carrying is None:
                continue
            pkg = agent.carrying
            if pkg.status == TaskStatus.IN_TRANSIT:
                dist = math.sqrt(
                    (agent.pos[0] - pkg.destination[0]) ** 2
                    + (agent.pos[1] - pkg.destination[1]) ** 2
                )
                if dist < 10.0:
                    pkg.status   = TaskStatus.DELIVERED
                    agent.carrying = None
                    agent.waypoints = []
            elif pkg.status == TaskStatus.ASSIGNED:
                # Agent moving toward origin
                dist = math.sqrt(
                    (agent.pos[0] - pkg.origin[0]) ** 2
                    + (agent.pos[1] - pkg.origin[1]) ** 2
                )
                if dist < 10.0:
                    pkg.status = TaskStatus.IN_TRANSIT

        # Mark overdue packages as failed
        for pkg in self._packages:
            if pkg.status == TaskStatus.PENDING and self._time > pkg.deadline:
                pkg.status = TaskStatus.FAILED