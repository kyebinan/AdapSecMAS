# env/network_env.py
# =============================================================================
# AdapSecMAS — NetworkEnv
# Facade pattern: single entry point for the POSG training environment.
# Hides the complexity of: channel, attackers, protocols, SLM, reward, gossip.
# The MARL trainer only ever calls reset() and step().
#
# POSG: G = <I, S, {A_i}, T, R_team, {O_i}, gamma>
#   I        = 20 agents
#   S        = channel state x attack state (latent)
#   A_i      = {0..6} discrete actions
#   T        = channel transitions + attacker movement
#   R_team   = RewardComputer output
#   O_i      = local noisy observation (7D own + 5D agg_comm)
#   gamma    = 0.99
#
# SRP: orchestrates the simulation step — delegates all logic to sub-modules.
# =============================================================================

from __future__ import annotations

import random
import numpy as np
from collections import defaultdict, deque

from core.interfaces  import IMetricsObserver
from core.message     import Message, MessageType, PeerMessage
from core.metrics     import StepMetrics
from core.constants   import (
    N_AGENTS, DIM_OBS_TOTAL, DIM_OBS, DIM_COMM,
    ARENA_WIDTH, ARENA_HEIGHT, DT_HEADLESS,
    TX_POWER_DEFAULT, TX_MULTIPLIERS,
    SNR_THRESHOLD, SNR_MAX_NORM,
    FLOOD_RATE_ATTACK, FLOOD_RATE_THRESHOLD,
    QUEUE_MAX,
)

from network.channel             import WirelessChannel
from network.gossip              import GossipMediator
from network.validation_pipeline import build_validation_pipeline

from attackers.factory import AttackerFactory
from attackers.jammer  import JammerAgent
from attackers.flooder import FloodAgent
from attackers.spoofer import SpoofAgent

from protocols.factory import ProtocolFactory

from security.levels       import SecurityLevel
from security.state_machine import SecurityLevelMachine
from security.reward        import RewardComputer


class NetworkEnv:
    """
    Cooperative POSG training environment — network security layer only.

    Facade: the MARL trainer calls reset() and step() only.
    All physical, protocol, and security logic is delegated internally.

    Observation per agent: np.ndarray of shape (DIM_OBS_TOTAL,) = (12,)
      [0]  snr_norm
      [1]  jammer_noise_norm
      [2]  corrupt_rate
      [3]  flood_rate_norm
      [4]  spoof_flag
      [5]  tx_norm
      [6]  level_norm
      [7-11] agg_comm (5D peer message aggregate)
    """

    # Number of recent messages tracked for corrupt_rate computation
    _CORRUPT_WINDOW: int = 5

    def __init__(
        self,
        n_agents    : int   = N_AGENTS,
        dt          : float = DT_HEADLESS,
        arena_w     : float = ARENA_WIDTH,
        arena_h     : float = ARENA_HEIGHT,
        rng_seed    : int | None = None,
    ):
        self._n       = n_agents
        self._dt      = dt
        self._arena_w = arena_w
        self._arena_h = arena_h
        self._rng     = random.Random(rng_seed)
        self._np_rng  = np.random.default_rng(rng_seed)

        # -- sub-modules (Facade hides these from the trainer) --
        self._channel   = WirelessChannel(rng_seed=rng_seed)
        self._gossip    = GossipMediator(n_agents)
        self._reward_fn = RewardComputer()
        self._protocols = ProtocolFactory.create_default_set()

        # -- attackers --
        self._attackers = AttackerFactory.create_default_set(rng_seed=rng_seed)
        self._jammer    = self._attackers[0]   # JammerAgent
        self._flooder   = self._attackers[1]   # FloodAgent
        self._spoofer   = self._attackers[2]   # SpoofAgent
        for a in self._attackers:
            self._channel.add_attacker(a)

        # -- per-agent state --
        self._positions  : dict[int, tuple[float, float]] = {}
        self._tx_powers  : dict[int, float]               = {}
        self._slms       : dict[int, SecurityLevelMachine] = {
            i: SecurityLevelMachine() for i in range(n_agents)
        }
        self._seq_out    : dict[int, int]   = {}   # outgoing seq counter per agent
        self._public_keys: dict[int, int]   = {}
        self._queues     : dict[int, int]   = {}   # current queue length

        # Sliding windows for corrupt_rate and flood_rate
        self._recv_windows  : dict[int, deque] = {}   # (success/fail per msg)
        self._flood_windows : dict[int, deque] = {}   # msg count per step from suspects

        # Spoof detection log: {agent_id: [(forged_id, seq_delta), ...]}
        self._spoof_log: dict[int, list] = defaultdict(list)

        # Shared network state (read/written by protocols)
        self._network_state: dict = {}

        # -- observers (Observer pattern) --
        self._observers: list[IMetricsObserver] = []

        # -- episode state --
        self._step_count      : int   = 0
        self._prev_delivery   : float = 0.0
        self._prev_proto_ok   : float = 0.0
        self._metrics         : StepMetrics = StepMetrics()

        # -- validation pipeline --
        self._pipeline = None   # built in reset() after keys are assigned

    # ------------------------------------------------------------------
    # Observer management
    # ------------------------------------------------------------------

    def subscribe(self, observer: IMetricsObserver) -> None:
        self._observers.append(observer)

    def _notify_step(self) -> None:
        for obs in self._observers:
            obs.on_step(self._metrics)

    # ------------------------------------------------------------------
    # Public API — Facade interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[int, np.ndarray]:
        """
        Reset the environment to the initial state.
        Returns initial observations per agent.
        """
        if seed is not None:
            self._rng    = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self._step_count    = 0
        self._prev_delivery = 0.0
        self._prev_proto_ok = 0.0
        self._spoof_log     = defaultdict(list)

        # Initialise agent positions (random scatter in arena)
        self._positions = {
            i: (
                self._rng.uniform(50, self._arena_w - 50),
                self._rng.uniform(50, self._arena_h - 50),
            )
            for i in range(self._n)
        }

        # Reset tx_power, queues, windows
        self._tx_powers = {i: TX_POWER_DEFAULT for i in range(self._n)}
        self._queues    = {i: 0 for i in range(self._n)}
        self._seq_out   = {i: 0 for i in range(self._n)}
        self._recv_windows  = {i: deque(maxlen=self._CORRUPT_WINDOW) for i in range(self._n)}
        self._flood_windows = {i: deque(maxlen=10) for i in range(self._n)}

        # Assign public/private keys (symmetric for simulation)
        self._public_keys = {i: (i + 1) * 1000 for i in range(self._n)}

        # Rebuild validation pipeline with fresh keys
        self._pipeline = build_validation_pipeline(self._public_keys)

        # Reset SLMs
        for slm in self._slms.values():
            slm.reset()

        # Reset gossip inbox
        self._gossip.reset()

        # Reset shared network state
        self._network_state = {
            "channel"              : {i: 0 for i in range(self._n)},
            "tx_power"             : self._tx_powers,
            "snr"                  : {},
            "flood_rate_per_sender": {},
            "banned"               : {},
            "crl"                  : set(),
            "revoked_steps"        : {},
            "spoof_log"            : self._spoof_log,
            "legitimate_ids"       : set(range(self._n)),
        }

        # Reset attacker positions
        for attacker in self._attackers:
            attacker.step(0.0)

        return self._build_observations()

    def step(
        self,
        actions: dict[int, int],
    ) -> tuple[dict[int, np.ndarray], float, bool, dict]:
        """
        Advance the simulation by one step.

        Parameters
        ----------
        actions : {agent_id: action_id}  joint action from all agents

        Returns
        -------
        obs     : {agent_id: np.ndarray}  new observations
        reward  : float                   team reward R_team
        done    : bool                    episode termination flag
        info    : dict                    step metrics for logging
        """
        self._step_count += 1
        self._metrics     = StepMetrics(
            step     = self._step_count,
            sim_time = self._step_count * self._dt,
        )

        # 1. Apply agent actions
        levels = self._apply_actions(actions)

        # 2. Step attackers
        self._step_attackers()

        # 3. Simulate message exchange
        self._simulate_communications()

        # 4. Decrement bans
        self._tick_bans()

        # 5. Run protocols triggered by action=6
        self._run_protocols(actions, levels)

        # 6. Broadcast peer messages (m_i_t)
        self._broadcast_peer_messages(levels)

        # 7. Build observations
        obs = self._build_observations()

        # 8. Compute reward
        reward = self._reward_fn.compute(self._metrics, actions, levels)

        # 9. Update SLMs with new observations
        for i in range(self._n):
            self._slms[i].update(obs[i][:DIM_OBS])

        # 10. Notify observers
        self._notify_step()

        done = False   # episodic termination managed by train.py
        return obs, reward, done, self._metrics.to_dict()

    # ------------------------------------------------------------------
    # Step internals
    # ------------------------------------------------------------------

    def _apply_actions(self, actions: dict[int, int]) -> dict[int, SecurityLevel]:
        """
        Apply environment actions (X_i):
          0: noop
          1: boost x8
          2: boost x16
          3: rate-limit  (handled in _simulate_communications)
          4: quarantine  (marks peer as ignored)
          5: verify-nonce (activates seq check)
          6: trigger protocol (handled in _run_protocols)
        Returns current security level per agent.
        """
        levels = {}
        for i, action in actions.items():
            # Update tx_power from multiplier
            mult               = TX_MULTIPLIERS.get(action, 1.0)
            self._tx_powers[i] = TX_POWER_DEFAULT * mult
            self._network_state["tx_power"][i] = self._tx_powers[i]

            # Handle quarantine (action=4) — mark suspect in network_state
            if action == 4:
                self._network_state.setdefault("quarantined_by", {})
                # quarantine the flooder agent if detected
                flooder_id = getattr(self._flooder, 'agent_id', None)
                if flooder_id is not None:
                    self._network_state["quarantined_by"].setdefault(i, set())
                    self._network_state["quarantined_by"][i].add(flooder_id)

            # Current level from SLM
            levels[i] = self._slms[i].level

        # Update level distribution metrics
        self._metrics.n_agents_normal   = sum(1 for l in levels.values() if l == SecurityLevel.NORMAL)
        self._metrics.n_agents_elevated = sum(1 for l in levels.values() if l == SecurityLevel.ELEVATED)
        self._metrics.n_agents_high     = sum(1 for l in levels.values() if l == SecurityLevel.HIGH)
        self._metrics.n_agents_critical = sum(1 for l in levels.values() if l == SecurityLevel.CRITICAL)

        # Level mismatch: mean gap between threat level and SLM level
        threat_levels = self._network_state.get("threat_levels", {i: 0 for i in range(self._n)})
        mismatches = [
            max(0, threat_levels.get(i, 0) - int(levels[i]))
            for i in range(self._n)
        ]
        self._metrics.level_mismatch = float(np.mean(mismatches))

        return levels

    def _step_attackers(self) -> None:
        """Move all attackers and update SNR map in network_state."""
        for attacker in self._attackers:
            attacker.step(self._dt, self._arena_w, self._arena_h)

        # Update per-link SNR in shared state
        snr_map = {}
        for i in range(self._n):
            snr_map[i] = self._channel.snr(
                self._positions[i],
                self._positions[i],   # own noise level
                self._tx_powers[i],
            )
            # Cross-link SNR for protocol use
            for j in range(self._n):
                if i != j:
                    snr_map[(i, j)] = self._channel.snr(
                        self._positions[i],
                        self._positions[j],
                        self._tx_powers[i],
                    )
        self._network_state["snr"] = snr_map

    def _simulate_communications(self) -> None:
        """
        Simulate one round of message exchange.
        Each agent sends one coordination message to a random neighbour.
        Flooder injects extra messages.
        All messages pass through the validation pipeline.
        """
        self._metrics.n_links_total = self._n * (self._n - 1)
        sent = delivered = lost_jam = corrupted = 0

        # -- Legitimate messages --
        for sender_id in range(self._n):
            # Pick a random receiver
            receiver_id = self._rng.choice(
                [j for j in range(self._n) if j != sender_id]
            )

            msg = self._build_message(sender_id)
            result, snr_val = self._channel.deliver(
                msg,
                self._positions[sender_id],
                self._positions[receiver_id],
                self._tx_powers[sender_id],
            )
            sent += 1

            if result is None:
                lost_jam += 1
                self._recv_windows[receiver_id].append(0)
            else:
                validation = self._pipeline.handle(result, receiver_id)
                if validation.valid:
                    delivered += 1
                    self._recv_windows[receiver_id].append(1)
                    self._update_snr_metric(snr_val)
                else:
                    corrupted += 1
                    self._recv_windows[receiver_id].append(0)
                    if validation.handler == "SeqCounterHandler":
                        self._record_spoof(receiver_id, result)

        # -- Flood messages --
        flood_msgs = self._flooder.generate_flood_messages(self._dt, self._step_count * self._dt)
        for fmsg in flood_msgs:
            victim_id = self._flooder.victim_id
            if victim_id is None:
                victim_id = self._rng.randint(0, self._n - 1)

            # Track flood rate per sender for BanVote
            fd_key = self._flooder.agent_id
            self._network_state["flood_rate_per_sender"][fd_key] = (
                self._network_state["flood_rate_per_sender"].get(fd_key, 0) + 1
            )

            # Add to victim queue
            if self._queues[victim_id] < QUEUE_MAX:
                self._queues[victim_id] += 1
            else:
                self._metrics.n_queue_overflows += 1

        self._metrics.n_msgs_sent        = sent
        self._metrics.n_msgs_delivered   = delivered
        self._metrics.n_msgs_lost_to_jam = lost_jam
        self._metrics.n_msgs_corrupted   = corrupted

        # Delivery delta
        current_rate              = delivered / max(sent, 1)
        self._metrics.delta_delivery_rate = current_rate - self._prev_delivery
        self._prev_delivery       = current_rate

        # Healthy links
        self._metrics.n_links_healthy = sum(
            1 for i in range(self._n)
            if self._network_state["snr"].get(i, 0) >= SNR_THRESHOLD
        )

    def _run_protocols(
        self,
        actions: dict[int, int],
        levels : dict[int, SecurityLevel],
    ) -> None:
        """Agents that chose action=6 trigger the appropriate protocol."""
        triggered = success = failed = 0

        for agent_id, action in actions.items():
            if action != 6:
                continue

            obs_agent  = self._build_single_obs(agent_id, levels[agent_id])
            neighbours = self._get_neighbours(agent_id)
            protocol   = ProtocolFactory.select(self._protocols, obs_agent[:DIM_OBS])

            if protocol is None:
                continue

            triggered += 1
            result = protocol.execute(agent_id, neighbours, self._network_state)

            if result.success:
                success += 1
            else:
                failed += 1

        proto_rate = success / max(triggered, 1)
        self._metrics.n_protocol_triggered  = triggered
        self._metrics.n_protocol_success    = success
        self._metrics.n_protocol_failed     = failed
        self._metrics.delta_protocol_success = proto_rate - self._prev_proto_ok
        self._prev_proto_ok                  = proto_rate

    def _broadcast_peer_messages(self, levels: dict[int, SecurityLevel]) -> None:
        """Each agent broadcasts m_i_t = f(o_t^i) — deterministic, not learned."""
        for i in range(self._n):
            corrupt_rate = self._compute_corrupt_rate(i)
            flood_rate   = self._compute_flood_rate(i)
            snr_norm     = self._compute_snr_norm(i)

            pm = PeerMessage(
                sender_id    = i,
                snr_norm     = snr_norm,
                corrupt_rate = corrupt_rate,
                flood_rate   = flood_rate,
                spoof_flag   = 1.0 if self._spoof_log[i] else 0.0,
                level_norm   = self._slms[i].level_norm(),
            )

            self._gossip.broadcast(
                sender_id = i,
                message   = pm,
                positions = self._positions,
                tx_powers = self._tx_powers,
                channel   = self._channel,
            )

    def _build_observations(self) -> dict[int, np.ndarray]:
        """Build observation vector for all agents."""
        levels = {i: self._slms[i].level for i in range(self._n)}
        return {
            i: self._build_single_obs(i, levels[i])
            for i in range(self._n)
        }

    def _build_single_obs(
        self,
        agent_id: int,
        level   : SecurityLevel,
    ) -> np.ndarray:
        """
        Build o_t^i ∈ R^12.
        [0..6] own obs, [7..11] agg_comm from peers.
        """
        snr_norm     = self._compute_snr_norm(agent_id)
        jammer_noise = self._channel.jammer_noise_at(self._positions[agent_id])
        jammer_norm  = float(np.clip(jammer_noise / 100.0, 0.0, 1.0))
        corrupt_rate = self._compute_corrupt_rate(agent_id)
        flood_rate   = self._compute_flood_rate(agent_id)
        spoof_flag   = 1.0 if self._spoof_log[agent_id] else 0.0
        tx_norm      = self._tx_powers[agent_id] / (TX_POWER_DEFAULT * 16.0)
        level_norm   = self._slms[agent_id].level_norm()

        own_obs = np.array([
            snr_norm, jammer_norm, corrupt_rate,
            flood_rate, spoof_flag, tx_norm, level_norm,
        ], dtype=np.float32)

        # Aggregate peer messages
        peer_msgs  = self._gossip.collect(agent_id)
        agg_comm   = self._gossip.aggregate(peer_msgs)

        return np.concatenate([own_obs, agg_comm]).astype(np.float32)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_message(self, sender_id: int) -> Message:
        """Build a signed coordination message from sender_id."""
        seq  = self._seq_out[sender_id]
        self._seq_out[sender_id] += 1
        msg  = Message(
            sender_id = sender_id,
            msg_type  = MessageType.TASK_REQUEST,
            seq       = seq,
            payload   = {},
            timestamp = self._step_count * self._dt,
        )
        return msg.sign(self._public_keys[sender_id])

    def _get_neighbours(self, agent_id: int) -> list[int]:
        """Return agents within communication range (SNR >= threshold)."""
        return [
            j for j in range(self._n)
            if j != agent_id
            and self._network_state["snr"].get((agent_id, j), 0) >= SNR_THRESHOLD
        ]

    def _compute_snr_norm(self, agent_id: int) -> float:
        """Normalised SNR at agent position — proxy for jamming intensity."""
        raw = self._channel.snr(
            self._positions[agent_id],
            self._positions[agent_id],
            self._tx_powers[agent_id],
        )
        return float(np.clip(raw / SNR_MAX_NORM, 0.0, 1.0))

    def _compute_corrupt_rate(self, agent_id: int) -> float:
        """Fraction of recent messages that failed — sliding window."""
        window = self._recv_windows[agent_id]
        if not window:
            return 0.0
        return 1.0 - (sum(window) / len(window))

    def _compute_flood_rate(self, agent_id: int) -> float:
        """Normalised flood rate observed by agent_id."""
        flooder_id = getattr(self._flooder, 'agent_id', -1)
        raw_rate   = self._network_state["flood_rate_per_sender"].get(flooder_id, 0.0)
        return float(np.clip(raw_rate / FLOOD_RATE_ATTACK, 0.0, 1.0))

    def _update_snr_metric(self, snr_val: float) -> None:
        """Update running SNR average in metrics."""
        n = self._metrics.n_msgs_delivered
        if n == 0:
            self._metrics.avg_snr = snr_val
        else:
            self._metrics.avg_snr = (self._metrics.avg_snr * (n - 1) + snr_val) / n

    def _record_spoof(self, receiver_id: int, message: Message) -> None:
        """Record a detected spoof attempt in the log."""
        self._metrics.n_spoof_attempts += 1
        expected = self._seq_out.get(message.sender_id, 0)
        delta    = abs(message.seq - expected)
        self._spoof_log[receiver_id].append((message.sender_id, delta))

        # Check if it was accepted before detection
        if message.payload.get("__forged__"):
            self._metrics.n_spoof_accepted += 1
        else:
            self._metrics.n_spoof_blocked += 1

    def _tick_bans(self) -> None:
        """Decrement ban counters — remove expired bans."""
        banned = self._network_state.get("banned", {})
        expired = [sid for sid, steps in banned.items() if steps <= 1]
        for sid in expired:
            del banned[sid]
        for sid in banned:
            banned[sid] -= 1

    # ------------------------------------------------------------------
    # Properties (read-only access for eval / production)
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        return self._n

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return (DIM_OBS_TOTAL,)

    @property
    def n_actions(self) -> int:
        from core.constants import N_ACTIONS
        return N_ACTIONS

    @property
    def network_state(self) -> dict:
        return self._network_state

    @property
    def positions(self) -> dict[int, tuple[float, float]]:
        return self._positions