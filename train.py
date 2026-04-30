# train.py
# =============================================================================
# AdapSecMAS — Headless training loop
# No rendering — pure network simulation.
# Curriculum: jam only → jam + flood → all 3 attacks
#
# Usage:
#   python train.py
#   python train.py --episodes 500 --seed 42 --no-curriculum
#
# Outputs:
#   checkpoints/best.pt       best policy weights
#   checkpoints/lastN.pt      checkpoint every N episodes
#   logs/train_steps.csv      per-step metrics
#   logs/train_episodes.csv   per-episode summary
# =============================================================================

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from simulation.network_env import NetworkEnv
from agents.mappo           import MAPPOTrainer
from core.constants         import N_AGENTS, UPDATE_EVERY
from core.metrics           import EpisodeSummary


# ---------------------------------------------------------------------------
# Curriculum configuration
# ---------------------------------------------------------------------------

@dataclass
class CurriculumPhase:
    name          : str
    episode_start : int
    jam_active    : bool
    flood_active  : bool
    spoof_active  : bool


CURRICULUM: list[CurriculumPhase] = [
    CurriculumPhase("jam_only",  episode_start=0,   jam_active=True,  flood_active=False, spoof_active=False),
    CurriculumPhase("jam_flood", episode_start=50,  jam_active=True,  flood_active=True,  spoof_active=False),
    CurriculumPhase("all",       episode_start=120, jam_active=True,  flood_active=True,  spoof_active=True),
]


def current_phase(episode: int) -> CurriculumPhase:
    """Return the active curriculum phase for the given episode."""
    phase = CURRICULUM[0]
    for p in CURRICULUM:
        if episode >= p.episode_start:
            phase = p
    return phase


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_dirs() -> None:
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)


def make_csv_writer(path: str, fieldnames: list[str]):
    """Open a CSV file and return (file_handle, writer)."""
    fh = open(path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    setup_dirs()

    # -- CSV loggers --
    step_fields = [
        "episode", "step", "reward",
        "n_msgs_sent", "n_msgs_delivered", "n_msgs_lost_to_jam",
        "n_queue_overflows", "n_spoof_accepted",
        "n_protocol_triggered", "n_protocol_success",
        "n_agents_normal", "n_agents_elevated",
        "n_agents_high", "n_agents_critical",
        "avg_snr", "phase",
    ]
    ep_fields = [
        "episode", "total_reward", "mean_delivery_rate",
        "total_msgs_lost_to_jam", "total_queue_overflows",
        "total_spoof_accepted", "protocol_success_rate",
        "pct_normal", "pct_elevated", "pct_high", "pct_critical",
        "duration_s", "phase",
    ]

    step_fh, step_writer = make_csv_writer("logs/train_steps.csv",   step_fields)
    ep_fh,   ep_writer   = make_csv_writer("logs/train_episodes.csv", ep_fields)

    # -- Environment + Trainer --
    env     = NetworkEnv(n_agents=N_AGENTS, rng_seed=args.seed)
    trainer = MAPPOTrainer(device="auto")

    best_reward  = float("-inf")
    global_step  = 0

    print(f"\nAdapSecMAS training")
    print(f"  episodes : {args.episodes}")
    print(f"  device   : {trainer.device}")
    print(f"  curriculum: {not args.no_curriculum}")
    print(f"  seed     : {args.seed}\n")

    for episode in range(1, args.episodes + 1):
        phase     = current_phase(episode) if not args.no_curriculum else CURRICULUM[-1]
        ep_start  = time.time()

        # -- Configure attackers for this phase --
        env._jammer._active  = phase.jam_active
        env._flooder._active = phase.flood_active
        env._spoofer._active = phase.spoof_active

        obs = env.reset(seed=args.seed + episode)
        trainer.reset_hiddens()

        ep_reward    = 0.0
        ep_steps     = 0
        step_metrics = []

        done = False
        while not done:
            # Act
            actions, log_probs, masks = trainer.act(obs)

            # Step environment
            obs_new, reward, done, info = env.step(actions)

            # Record in buffer
            trainer.record(obs, actions, log_probs, reward, done, masks)
            obs = obs_new

            ep_reward   += reward
            ep_steps    += 1
            global_step += 1

            # Log step
            step_row = {
                "episode"             : episode,
                "step"                : ep_steps,
                "reward"              : round(reward, 4),
                "n_msgs_sent"         : info["n_msgs_sent"],
                "n_msgs_delivered"    : info["n_msgs_delivered"],
                "n_msgs_lost_to_jam"  : info["n_msgs_lost_to_jam"],
                "n_queue_overflows"   : info["n_queue_overflows"],
                "n_spoof_accepted"    : info["n_spoof_accepted"],
                "n_protocol_triggered": info["n_protocol_triggered"],
                "n_protocol_success"  : info["n_protocol_success"],
                "n_agents_normal"     : info["n_agents_normal"],
                "n_agents_elevated"   : info["n_agents_elevated"],
                "n_agents_high"       : info["n_agents_high"],
                "n_agents_critical"   : info["n_agents_critical"],
                "avg_snr"             : round(info["avg_snr"], 4),
                "phase"               : phase.name,
            }
            step_writer.writerow(step_row)
            step_metrics.append(info)

            # PPO update when buffer is ready
            if trainer.should_update():
                update_metrics = trainer.update(last_obs=obs)
                _log_update(episode, ep_steps, update_metrics)

            # Episode termination — fixed length
            if ep_steps >= args.steps_per_episode:
                done = True

        # -- Episode summary --
        duration   = time.time() - ep_start
        total_sent = sum(m["n_msgs_sent"]      for m in step_metrics)
        total_del  = sum(m["n_msgs_delivered"] for m in step_metrics)
        total_jam  = sum(m["n_msgs_lost_to_jam"] for m in step_metrics)
        total_over = sum(m["n_queue_overflows"]  for m in step_metrics)
        total_spf  = sum(m["n_spoof_accepted"]   for m in step_metrics)
        total_trig = sum(m["n_protocol_triggered"] for m in step_metrics)
        total_succ = sum(m["n_protocol_success"]   for m in step_metrics)

        mean_dr   = total_del / max(total_sent, 1)
        proto_sr  = total_succ / max(total_trig, 1)

        total_level_steps = sum(
            m["n_agents_normal"] + m["n_agents_elevated"]
            + m["n_agents_high"] + m["n_agents_critical"]
            for m in step_metrics
        )
        pct_normal   = sum(m["n_agents_normal"]   for m in step_metrics) / max(total_level_steps, 1)
        pct_elevated = sum(m["n_agents_elevated"]  for m in step_metrics) / max(total_level_steps, 1)
        pct_high     = sum(m["n_agents_high"]      for m in step_metrics) / max(total_level_steps, 1)
        pct_critical = sum(m["n_agents_critical"]  for m in step_metrics) / max(total_level_steps, 1)

        ep_row = {
            "episode"           : episode,
            "total_reward"      : round(ep_reward, 2),
            "mean_delivery_rate": round(mean_dr,   4),
            "total_msgs_lost_to_jam": total_jam,
            "total_queue_overflows" : total_over,
            "total_spoof_accepted"  : total_spf,
            "protocol_success_rate" : round(proto_sr, 4),
            "pct_normal"   : round(pct_normal,   4),
            "pct_elevated" : round(pct_elevated,  4),
            "pct_high"     : round(pct_high,      4),
            "pct_critical" : round(pct_critical,  4),
            "duration_s"   : round(duration, 2),
            "phase"        : phase.name,
        }
        ep_writer.writerow(ep_row)

        # Console output
        print(
            f"ep {episode:>4}/{args.episodes}  "
            f"[{phase.name:<10}]  "
            f"reward={ep_reward:>8.1f}  "
            f"delivery={mean_dr:.2%}  "
            f"proto_sr={proto_sr:.2%}  "
            f"t={duration:.1f}s"
        )

        # Save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            trainer.save("checkpoints/best.pt")

        # Periodic checkpoint
        if episode % args.checkpoint_every == 0:
            trainer.save(f"checkpoints/ep{episode:04d}.pt")

    step_fh.close()
    ep_fh.close()
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print("Weights: checkpoints/best.pt")
    print("Logs   : logs/train_episodes.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_update(episode: int, step: int, metrics: dict) -> None:
    print(
        f"  [update ep={episode} step={step}]  "
        f"p_loss={metrics['policy_loss']:.4f}  "
        f"v_loss={metrics['value_loss']:.4f}  "
        f"entropy={metrics['entropy']:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdapSecMAS — MAPPO training")
    parser.add_argument("--episodes",          type=int,   default=200)
    parser.add_argument("--steps-per-episode", type=int,   default=512,
                        dest="steps_per_episode")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--checkpoint-every",  type=int,   default=50,
                        dest="checkpoint_every")
    parser.add_argument("--no-curriculum",     action="store_true",
                        dest="no_curriculum")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)