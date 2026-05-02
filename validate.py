"""Simple CLI to validate a trained model on predefined routes.

Writes a TSV with per-episode results across `config.VALIDATION_ROUTES` and a sweep
of traffic scales.
"""

from __future__ import annotations

import argparse

from config import SUMO_CONFIG
from utils import validate_model_on_routes




def main() -> None:
    scales = [2, 3, 5]

    out_path = validate_model_on_routes(
        model_path="dqn_training_4/dqn_ego_episode_500.pth",
        sumo_config=SUMO_CONFIG,
        traffic_scales=scales,
        use_gui=False,
        max_steps=1500,
        out_tsv_path='./dqn_training_4/validation_results.tsv',
    )

    print(out_path)


if __name__ == "__main__":
    main()
