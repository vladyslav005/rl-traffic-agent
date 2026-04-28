"""Ego-related event accounting helpers.

Why this exists
---------------
TraCI collision/teleport/emergency APIs can report the same incident across
multiple simulation steps. If you simply do `counter += len(events)` every step,
counts become inflated.

This module provides a tiny stateful counter that:
- de-duplicates collision events across steps within an episode
- counts ego-related incidents *exactly once per episode* for "crashes"
- counts ego collisions as *events* (can be >1 per episode)
- counts ego teleports/emergency stops at most once per episode

You can reset the state at the start of every episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Set, Tuple


def _collision_key(collision: Any, sim_time: float) -> Hashable:
    """Return a stable key for a TraCI Collision object.

    Newer SUMO versions provide `collisionID`. If not available, we fall back to
    a tuple based on time + collider/victim.
    """
    cid = getattr(collision, "collisionID", None)
    if cid is not None:
        return cid

    return (
        sim_time,
        getattr(collision, "collider", None),
        getattr(collision, "victim", None),
    )


@dataclass
class EgoEventState:
    seen_collision_keys: Set[Hashable] = field(default_factory=set)
    ego_incident_counted: bool = False
    ego_teleport_counted: bool = False
    ego_emergency_counted: bool = False


@dataclass
class EgoEventDelta:
    """How much to increment each counter for exactly one simulation step."""

    total_collision_events: int = 0
    total_ego_collisions: int = 0
    total_ego_crashes: int = 0
    total_ego_teleports: int = 0
    total_ego_emergency_stops: int = 0


def accumulate_ego_events(
    *,
    state: EgoEventState,
    ego_id: str,
    sim_time: float,
    collisions: list[Any],
    teleport_ids: list[str],
    emergency_ids: list[str],
) -> Tuple[EgoEventState, EgoEventDelta]:
    """Accumulate per-step TraCI events into correct, de-duplicated counters.

    Returns updated (state, delta). Add `delta.*` to your global counters.
    """

    delta = EgoEventDelta()

    # Collisions (dedup within episode)
    for c in collisions:
        key = _collision_key(c, sim_time)
        if key in state.seen_collision_keys:
            continue
        state.seen_collision_keys.add(key)

        delta.total_collision_events += 1

        collider = getattr(c, "collider", None)
        victim = getattr(c, "victim", None)
        if ego_id in (collider, victim):
            delta.total_ego_collisions += 1
            if not state.ego_incident_counted:
                delta.total_ego_crashes += 1
                state.ego_incident_counted = True

    # Teleports (count once per episode)
    if (ego_id in teleport_ids) and (not state.ego_teleport_counted):
        delta.total_ego_teleports += 1
        state.ego_teleport_counted = True
        if not state.ego_incident_counted:
            delta.total_ego_crashes += 1
            state.ego_incident_counted = True

    # Emergency stops (count once per episode)
    if (ego_id in emergency_ids) and (not state.ego_emergency_counted):
        delta.total_ego_emergency_stops += 1
        state.ego_emergency_counted = True
        if not state.ego_incident_counted:
            delta.total_ego_crashes += 1
            state.ego_incident_counted = True

    return state, delta
