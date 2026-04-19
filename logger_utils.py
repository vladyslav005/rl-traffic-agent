from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class EpisodeLogger:
    """Append episode summaries to a log file.

    Writes one line per episode. Flushes on every write so you don't lose progress
    if SUMO or the kernel crashes.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "timestamp\tepisode\treward\tsteps\tend_reason\t"
                "TOTAL_EGO_CRASHES\tTOTAL_COLLISION_EVENTS\tTOTAL_EGO_COLLISIONS\t"
                "TOTAL_EGO_TELEPORTS\tTOTAL_EGO_EMERGENCY_STOPS\n",
                encoding="utf-8",
            )

    def log(
        self,
        *,
        episode: int,
        reward: float,
        steps: int,
        end_reason: str,
        total_ego_crashes: int,
        total_collision_events: int,
        total_ego_collisions: int,
        total_ego_teleports: int,
        total_ego_emergency_stops: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
        line = (
            f"{ts}\t{episode}\t{reward:.6f}\t{steps}\t{end_reason}\t"
            f"{total_ego_crashes}\t{total_collision_events}\t{total_ego_collisions}\t"
            f"{total_ego_teleports}\t{total_ego_emergency_stops}\n"
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def default_episode_log_path(*, base_dir: str | Path = "logs") -> Path:
    """Create a timestamped default path like logs/train_YYYYmmdd_HHMMSS.tsv."""

    base = Path(base_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"train_{ts}.tsv"
