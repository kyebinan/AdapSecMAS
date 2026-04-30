# eval.py
# =============================================================================
# AdapSecMAS — Evaluation script
# Runs 3 mandatory scenarios from the thesis (Section 3.5):
#
#   sim1 — baseline : no attacks, no defence  → measure M baseline
#   sim2 — attack   : all 3 attacks, no MARL  → measure degradation
#   sim3 — defended : all 3 attacks, MARL on  → measure recovery
#
# Usage:
#   python eval.py --weights checkpoints/best.pt
#   python eval.py --weights checkpoints/best.pt --episodes 30 --seed 0
#
# Outputs:
#   logs/eval_summary.csv     per-scenario aggregated metrics
#   logs/eval_episodes.csv    per-episode metrics for all scenarios
# =============================================================================

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, asdict

import numpy as np

from simulation.network_env import NetworkEnv
from agents.mappo           import MAPPOTrainer
from core.constants         import N_AGENTS


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name        : str
    jam_active  : bool
    flood_active: bool
    spoof_active: bool
    marl_active : bool
    description : str


SCENARIOS: list[Scenario] = [
    Scenario(
        name         = "sim1_baseline",
        jam_active   = False,
        flood_active = False,
        spoof_active = False,
        marl_active  = False,
        description  = "No attacks, no defence — establishes mission baseline",
    ),
    Scenario(
        name         = "sim2_attack",
        jam_active   = True,
        flood_active = True,
        spoof_active = True,
        marl_active  = False,
        description  = "All 3 attacks active, no defence — measures degradation",
    ),
    Scenario(
        name         = "sim3_defended",
        jam_active   = True,
        flood_active = True,
        spoof_active = True,
        marl_active  = True,
        description  = "All 3 attacks active, MARL defence on — measures recovery",
    ),
]


# ---------------------------------------------------------------------------
# Per-episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    scenario           : str
    episode            : int
    total_reward       : float
    mean_delivery_rate : float
    total_msgs_sent    : int
    total_msgs_delivered: int
    total_msgs_lost_jam: int
    total_queue_overflows: int
    total_spoof_accepted : int
    total_spoof_blocked  : int
    protocol_success_rate: float
    mean_snr             : float
    pct_normal           : float
    pct_elevated         : float
    pct_high             : float
    pct_critical         : float


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario  : Scenario,
    trainer   : MAPPOTrainer | None,
    n_episodes: int,
    steps_per : int,
    seed      : int,
) -> list[EpisodeResult]:
    """
    Run one scenario for n_episodes and return per-episode results.

    Parameters
    ----------
    trainer : MAPPOTrainer with loaded weights — None if marl_active=False
    """
    env     = NetworkEnv(n_agents=N_AGENTS, rng_seed=seed)
    results = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset(seed=seed + ep)

        # Configure attackers
        env._jammer._active  = scenario.jam_active
        env._flooder._active = scenario.flood_active
        env._spoofer._active = scenario.spoof_active

        if trainer is not None:
            trainer.reset_hiddens()

        ep_reward    = 0.0
        step_metrics = []

        for _ in range(steps_per):
            if scenario.marl_active and trainer is not None:
                actions, _, _ = trainer.act(obs)
            else:
                # Baseline / attack — all noop
                actions = {i: 0 for i in range(N_AGENTS)}

            obs, reward, _, info = env.step(actions)
            ep_reward += reward
            step_metrics.append(info)

        # Aggregate
        total_sent = sum(m["n_msgs_sent"]       for m in step_metrics)
        total_del  = sum(m["n_msgs_delivered"]  for m in step_metrics)
        total_jam  = sum(m["n_msgs_lost_to_jam"] for m in step_metrics)
        total_over = sum(m["n_queue_overflows"]   for m in step_metrics)
        total_spf  = sum(m["n_spoof_accepted"]    for m in step_metrics)
        total_blk  = sum(m["n_spoof_blocked"]     for m in step_metrics)
        total_trig = sum(m["n_protocol_triggered"] for m in step_metrics)
        total_succ = sum(m["n_protocol_success"]   for m in step_metrics)
        mean_snr   = np.mean([m["avg_snr"] for m in step_metrics])

        total_lvl  = sum(
            m["n_agents_normal"] + m["n_agents_elevated"]
            + m["n_agents_high"] + m["n_agents_critical"]
            for m in step_metrics
        )

        results.append(EpisodeResult(
            scenario             = scenario.name,
            episode              = ep,
            total_reward         = round(ep_reward, 2),
            mean_delivery_rate   = round(total_del / max(total_sent, 1), 4),
            total_msgs_sent      = total_sent,
            total_msgs_delivered = total_del,
            total_msgs_lost_jam  = total_jam,
            total_queue_overflows= total_over,
            total_spoof_accepted = total_spf,
            total_spoof_blocked  = total_blk,
            protocol_success_rate= round(total_succ / max(total_trig, 1), 4),
            mean_snr             = round(float(mean_snr), 4),
            pct_normal   = round(sum(m["n_agents_normal"]   for m in step_metrics) / max(total_lvl, 1), 4),
            pct_elevated = round(sum(m["n_agents_elevated"]  for m in step_metrics) / max(total_lvl, 1), 4),
            pct_high     = round(sum(m["n_agents_high"]      for m in step_metrics) / max(total_lvl, 1), 4),
            pct_critical = round(sum(m["n_agents_critical"]  for m in step_metrics) / max(total_lvl, 1), 4),
        ))

        print(
            f"  [{scenario.name}] ep {ep:>3}/{n_episodes}  "
            f"reward={ep_reward:>8.1f}  "
            f"delivery={total_del/max(total_sent,1):.2%}  "
            f"jam={total_jam}  overflow={total_over}  spoof={total_spf}"
        )

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise(results: list[EpisodeResult]) -> dict:
    """Compute mean ± std across episodes for one scenario."""
    def mean(field):
        vals = [getattr(r, field) for r in results]
        return float(np.mean(vals))

    def std(field):
        vals = [getattr(r, field) for r in results]
        return float(np.std(vals))

    return {
        "scenario"                  : results[0].scenario,
        "n_episodes"                : len(results),
        "mean_reward"               : round(mean("total_reward"),         2),
        "std_reward"                : round(std("total_reward"),          2),
        "mean_delivery_rate"        : round(mean("mean_delivery_rate"),   4),
        "std_delivery_rate"         : round(std("mean_delivery_rate"),    4),
        "mean_msgs_lost_jam"        : round(mean("total_msgs_lost_jam"),  2),
        "mean_queue_overflows"      : round(mean("total_queue_overflows"),2),
        "mean_spoof_accepted"       : round(mean("total_spoof_accepted"), 2),
        "mean_spoof_blocked"        : round(mean("total_spoof_blocked"),  2),
        "mean_protocol_success_rate": round(mean("protocol_success_rate"),4),
        "mean_snr"                  : round(mean("mean_snr"),             4),
        "pct_normal"                : round(mean("pct_normal"),           4),
        "pct_elevated"              : round(mean("pct_elevated"),         4),
        "pct_high"                  : round(mean("pct_high"),             4),
        "pct_critical"              : round(mean("pct_critical"),         4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    os.makedirs("logs", exist_ok=True)

    # Load trained policy
    trainer = None
    if args.weights:
        trainer = MAPPOTrainer(device="auto")
        trainer.load(args.weights)
        trainer.actor.eval()
        trainer.reset_hiddens()
        print(f"Loaded weights: {args.weights}\n")

    all_results  : list[EpisodeResult] = []
    all_summaries: list[dict]           = []

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  {scenario.name}")
        print(f"  {scenario.description}")
        print(f"{'='*60}")

        # sim3 requires loaded weights
        if scenario.marl_active and trainer is None:
            print("  [SKIP] sim3 requires --weights")
            continue

        results = run_scenario(
            scenario   = scenario,
            trainer    = trainer if scenario.marl_active else None,
            n_episodes = args.episodes,
            steps_per  = args.steps_per_episode,
            seed       = args.seed,
        )
        all_results.extend(results)

        summary = summarise(results)
        all_summaries.append(summary)

        print(f"\n  Summary [{scenario.name}]:")
        print(f"    delivery rate : {summary['mean_delivery_rate']:.2%} ± {summary['std_delivery_rate']:.2%}")
        print(f"    reward        : {summary['mean_reward']:.1f} ± {summary['std_reward']:.1f}")
        print(f"    msgs lost/jam : {summary['mean_msgs_lost_jam']:.1f}/ep")
        print(f"    queue overflow: {summary['mean_queue_overflows']:.1f}/ep")
        print(f"    spoof accepted: {summary['mean_spoof_accepted']:.1f}/ep")
        print(f"    proto success : {summary['mean_protocol_success_rate']:.2%}")

    # -- Write CSVs --
    ep_fields  = list(asdict(all_results[0]).keys()) if all_results else []
    sum_fields = list(all_summaries[0].keys())       if all_summaries else []

    with open("logs/eval_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields)
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))

    with open("logs/eval_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for s in all_summaries:
            w.writerow(s)

    print(f"\nResults saved:")
    print(f"  logs/eval_episodes.csv")
    print(f"  logs/eval_summary.csv")

    # -- Print comparison table --
    if len(all_summaries) >= 2:
        print(f"\n{'Scenario':<20} {'Delivery':>10} {'Lost/ep':>10} {'Spoof/ep':>10}")
        print("-" * 55)
        for s in all_summaries:
            print(
                f"{s['scenario']:<20} "
                f"{s['mean_delivery_rate']:>10.2%} "
                f"{s['mean_msgs_lost_jam']:>10.1f} "
                f"{s['mean_spoof_accepted']:>10.1f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdapSecMAS — evaluation")
    parser.add_argument("--weights",            type=str,   default=None,
                        help="Path to trained weights (.pt)")
    parser.add_argument("--episodes",           type=int,   default=30)
    parser.add_argument("--steps-per-episode",  type=int,   default=512,
                        dest="steps_per_episode")
    parser.add_argument("--seed",               type=int,   default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)