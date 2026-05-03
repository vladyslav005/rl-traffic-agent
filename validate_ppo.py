"""Simple CLI to validate a trained PPO model on predefined routes.

Mirrors `validate.py` (DQN) and writes a TSV with per-episode results across
`config.VALIDATION_ROUTES` and a sweep of traffic scales.

The TSV format matches DQN validation so you can compare models directly.
"""

from __future__ import annotations

import argparse

from config import SUMO_CONFIG
from utils import validate_ppo_model_on_routes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate a PPO checkpoint on validation routes")
    p.add_argument(
        "--model-path",
        default="ppo/ppo_training/ppo_ego_episode_500.pth",
        help="Path to PPO checkpoint (.pth) saved by ppo/train_ppo.py",
    )
    p.add_argument(
        "--sumo-config",
        default=SUMO_CONFIG,
        help="Path to SUMO .sumocfg (defaults to config.SUMO_CONFIG)",
    )
    p.add_argument(
        "--traffic-scales",
        default="2,3,5",
        help="Comma-separated list of traffic scale multipliers (e.g. 1,2,3)",
    )
    p.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    p.add_argument("--max-steps", type=int, default=1500, help="Max steps per episode")
    p.add_argument(
        "--out-tsv",
        default="ppo/ppo_training/validation_results.tsv",
        help="Output TSV path",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of argmax (not recommended for apples-to-apples)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    scales = [2, 3, 5]

    out_path = validate_ppo_model_on_routes(
        model_path="ppo/ppo_training/ppo_ego_episode_500.pth",
        sumo_config=SUMO_CONFIG,
        traffic_scales=scales,
        deterministic=not bool(args.stochastic),
        use_gui=False,
        max_steps=1500,
        out_tsv_path='ppo/validation_results_PPO.tsv',
    )

    print(out_path)


if __name__ == "__main__":
    main()
