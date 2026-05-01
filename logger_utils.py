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
                "timestamp\tepisode\treward\tsteps\tend_reason\troute\t"
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
        route_id: Optional[str] = None,
        total_ego_crashes: int,
        total_collision_events: int,
        total_ego_collisions: int,
        total_ego_teleports: int,
        total_ego_emergency_stops: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
        line = (
            f"{ts}\t{episode}\t{reward:.6f}\t{steps}\t{end_reason}\t{route_id}\t"
            f"{total_ego_crashes}\t{total_collision_events}\t{total_ego_collisions}\t"
            f"{total_ego_teleports}\t{total_ego_emergency_stops}\n"
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


@dataclass
class ValidationEpisodeLogger:
    """Append validation episode summaries to a TSV log file."""

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "timestamp\tcombo_episode\troute_id\ttraffic_scale\treward\tsteps\tend_reason\t"
                "ego_crash\tego_collision_events\tego_teleport\tego_emergency_stop\t"
                "sim_time_end\tmodel_path\tsumo_config\n",
                encoding="utf-8",
            )

    def log(
        self,
        *,
        combo_episode: int,
        route_id: str,
        traffic_scale: float,
        reward: float,
        steps: int,
        end_reason: str,
        ego_crash: int,
        ego_collision_events: int,
        ego_teleport: int,
        ego_emergency_stop: int,
        sim_time_end: float,
        model_path: str,
        sumo_config: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now()).isoformat(timespec="seconds")
        line = (
            f"{ts}\t{combo_episode}\t{route_id}\t{traffic_scale}\t{reward:.6f}\t{steps}\t{end_reason}\t"
            f"{ego_crash}\t{ego_collision_events}\t{ego_teleport}\t{ego_emergency_stop}\t"
            f"{sim_time_end:.1f}\t{model_path}\t{sumo_config}\n"
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def default_episode_log_path(*, base_dir: str | Path = "logs") -> Path:
    """Create a timestamped default path like logs/train_YYYYmmdd_HHMMSS.tsv."""

    base = Path(base_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"train_{ts}.tsv"


def default_validation_log_path(*, base_dir: str | Path = "logs") -> Path:
    """Create a timestamped default path like logs/validate_YYYYmmdd_HHMMSS.tsv."""

    base = Path(base_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"validate_{ts}.tsv"
