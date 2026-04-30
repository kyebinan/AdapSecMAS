# run_production.py
# =============================================================================
# AdapSecMAS — Production entry point
# Loads trained weights and runs the delivery simulation with two
# pygame windows side by side.
#
# Usage:
#   python run_production.py --weights checkpoints/best.pt
#   python run_production.py --weights checkpoints/best.pt --fps 30
# =============================================================================

from __future__ import annotations

import argparse
import sys

import pygame

from production.delivery_env import DeliveryEnv
from production.map_view     import MapView
from production.network_view import NetworkView


def run(args: argparse.Namespace) -> None:
    env      = DeliveryEnv(weights_path=args.weights, seed=args.seed)
    map_view = MapView(title="AdapSecMAS — City Map")
    net_view = NetworkView(title="AdapSecMAS — Network")

    state = env.reset()
    running = True

    print(f"Production started — weights: {args.weights}")
    print("Press ESC or close a window to quit.\n")

    while running:
        state = env.step()

        # Both windows must stay open
        if not map_view.render(state, fps=args.fps):
            running = False
        if not net_view.render(state, fps=args.fps):
            running = False

    map_view.close()
    net_view.close()
    print("Production ended.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdapSecMAS — production run")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (.pt)")
    parser.add_argument("--fps",     type=int, default=60)
    parser.add_argument("--seed",    type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)