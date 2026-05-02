from __future__ import annotations

import os
from typing import Optional

import numpy as np
import traci
from traci import TraCIException

from Action import ACTION_TO_DELTA_V
from config import *

CURRENT_USE_GUI = False
CURRENT_SUMO_CONFIG: str = SUMO_CONFIG
CURRENT_TRAFFIC_SCALE: float = 1.0

# Reward shaping note:
# Rewards here are computed *per simulation step*.
# If any positive per-step bonus is too large, the agent can exploit it by
# waiting/creeping ("reward farming") for hundreds of steps. Therefore:
# - per-step shaping bonuses must be small
# - the main positive signal should come from *real forward progress*
_LAST_DISTANCE_BY_VEH_ID: dict[str, float] = {}


def start_sumo(
               use_gui=False,
               log_dir="sim_logs",
               quiet_console=False,
               hide_warnings=False,
               sumo_config: Optional[str] = None,
               traffic_scale: Optional[float] = None,
):
    global CURRENT_USE_GUI, CURRENT_SUMO_CONFIG, CURRENT_TRAFFIC_SCALE
    CURRENT_USE_GUI = use_gui

    if sumo_config is not None:
        CURRENT_SUMO_CONFIG = sumo_config
    if traffic_scale is not None:
        CURRENT_TRAFFIC_SCALE = float(traffic_scale)

    os.makedirs(log_dir, exist_ok=True)

    binary = "sumo-gui" if use_gui else SUMO_BINARY
    cmd = [
        binary,
        "-c", CURRENT_SUMO_CONFIG,
        "--collision.action", "warn",
        "--collision.check-junctions", "true",
        "--collision.mingap-factor", "0",
        "--aggregate-warnings", "5",
        "--log", os.path.join(log_dir, "sumo.log"),
        "--log.timestamps", "true",
    ]

    # Traffic scaling (SUMO's --scale scales flows/vehicles; safe no-op if scenario has no flows)
    if CURRENT_TRAFFIC_SCALE != 1.0:
        cmd += ["--scale", str(CURRENT_TRAFFIC_SCALE)]

    if quiet_console:
        cmd += ["--no-step-log", "true"]

    if hide_warnings:
        cmd += ["--no-warnings", "true"]

    traci.start(cmd)


def _traci_is_connected() -> bool:
    """Best-effort check whether a TraCI connection is currently alive."""
    try:
        _ = traci.simulation.getTime()
        return True
    except Exception:
        return False


def reset_sumo(use_gui=None, *, sumo_config: Optional[str] = None, traffic_scale: Optional[float] = None):
    """Reset strategy: close current simulation and restart.

    Args:
        use_gui:
            - None: restart with the same mode as the current run (CURRENT_USE_GUI)
            - bool: explicitly restart in GUI/headless mode
        sumo_config:
            Optional override for the SUMO .sumocfg path.
        traffic_scale:
            Optional override for SUMO's `--scale` (float).
    """
    if use_gui is None:
        use_gui = CURRENT_USE_GUI

    # Close only if connection exists; avoids exceptions on first reset.
    if _traci_is_connected():
        try:
            traci.close()
        except Exception:
            pass

    # TraCI is restarted, so per-vehicle cached values (e.g., distance deltas)
    # must be reset as well.
    _LAST_DISTANCE_BY_VEH_ID.clear()

    start_sumo(use_gui=use_gui, sumo_config=sumo_config, traffic_scale=traffic_scale)


def spawn_ego(route_id, wait_steps=30):
    """
    Add ego once, then wait for insertion into the network.

    Returns:
        (success: bool, reason: str)
        reason in {"spawned", "invalid_route", "spawn_blocked", "spawn_error"}
    """
    # Clean up any previous ego that is still around
    try:
        if EGO_ID in traci.vehicle.getIDList():
            traci.vehicle.remove(EGO_ID)
    except Exception:
        pass

    depart_lane = "best"
    depart_speed = "0"
    depart_pos = "base"

    # Add only once
    try:
        traci.vehicle.add(
            vehID=EGO_ID,
            routeID=route_id,
            typeID=EGO_TYPE_ID,
            departLane=depart_lane,
            departSpeed=depart_speed,
            departPos=depart_pos,
        )
    except TraCIException as e:
        msg = str(e)

        if "has no valid route" in msg or "No connection between edge" in msg:
            return False, "invalid_route"

        if "already exists" in msg:
            # stale vehicle state; try to remove once and fail cleanly
            try:
                traci.vehicle.remove(EGO_ID)
            except Exception:
                pass
            return False, "spawn_error"

        return False, f"spawn_error: {msg}"
    except Exception as e:
        return False, f"spawn_error: {e}"

    # Wait for SUMO to actually insert the vehicle on the road
    for step in range(wait_steps):
        traci.simulationStep()

        if EGO_ID in traci.vehicle.getIDList():
            try:
                if CURRENT_USE_GUI:
                    traci.vehicle.setColor(EGO_ID, (255, 255, 255, 255))
            except Exception:
                pass

            # Only works in sumo-gui; harmless if unavailable
            try:
                if CURRENT_USE_GUI:
                    traci.gui.trackVehicle("View #0", EGO_ID)
            except Exception:
                pass

            # Initialize progress-tracking baseline for reward shaping.
            try:
                _LAST_DISTANCE_BY_VEH_ID[EGO_ID] = float(traci.vehicle.getDistance(EGO_ID))
            except Exception:
                _LAST_DISTANCE_BY_VEH_ID.pop(EGO_ID, None)

            return True, "spawned"

    # Vehicle never appeared on the road -> likely blocked insertion
    try:
        if EGO_ID in traci.vehicle.getIDList():
            traci.vehicle.remove(EGO_ID)
    except Exception:
        pass

    return False, "spawn_blocked"

def spawn_smpl(route_id):
    """
    Add ego once, then wait for insertion into the network.

    Returns:
        (success: bool, reason: str)
        reason in {"spawned", "invalid_route", "spawn_blocked", "spawn_error"}
    """
    # Clean up any previous ego that is still around

    depart_lane = "best"
    depart_speed = "0"
    depart_pos = "base"

    traci.vehicle.add(
        vehID=EGO_ID,
        routeID=route_id,
        typeID=EGO_TYPE_ID,
        departLane=depart_lane,
        departSpeed=depart_speed,
        departPos=depart_pos
    )


def ego_exists():
    return EGO_ID in traci.vehicle.getIDList()


def get_leader_info(veh_id, max_dist=100.0):
    leader = traci.vehicle.getLeader(veh_id, max_dist)
    if leader is None:
        return max_dist, 0.0
    leader_id, gap = leader
    leader_speed = traci.vehicle.getSpeed(leader_id)
    gap = min(gap, max_dist)
    return gap, leader_speed


def get_tls_info(veh_id, max_dist=200.0):
    """
    Returns:
        tls_dist_norm, is_red, is_green, is_yellow
    """
    tls_list = traci.vehicle.getNextTLS(veh_id)
    if not tls_list:
        return 1.0, 0.0, 0.0, 0.0

    # usually first entry is nearest next TLS
    tls_id, tls_index, tls_dist, tls_state = tls_list[0]

    tls_dist_norm = min(tls_dist, max_dist) / max_dist
    is_red = 1.0 if tls_state.lower() == "r" else 0.0
    is_green = 1.0 if tls_state.lower() == "g" else 0.0
    is_yellow = 1.0 if tls_state.lower() == "y" else 0.0

    return tls_dist_norm, is_red, is_green, is_yellow


def get_state(veh_id):
    ego_speed = traci.vehicle.getSpeed(veh_id)
    allowed_speed = traci.vehicle.getAllowedSpeed(veh_id)

    gap, leader_speed = get_leader_info(veh_id)
    rel_speed = ego_speed - leader_speed

    time_headway = gap / max(ego_speed, 0.1)
    time_headway = min(time_headway, 10.0)

    tls_dist, tls_red, tls_green, tls_yellow = get_tls_info(veh_id)

    lane_id = traci.vehicle.getLaneID(veh_id)
    lane_pos = traci.vehicle.getLanePosition(veh_id)
    lane_len = traci.lane.getLength(lane_id)
    lane_pos_norm = lane_pos / max(lane_len, 1.0)

    route = traci.vehicle.getRoute(veh_id)
    route_index = traci.vehicle.getRouteIndex(veh_id)
    route_progress = route_index / max(len(route) - 1, 1)

    state = np.array([
        ego_speed / 20.0,
        allowed_speed / 20.0,
        min(gap, 100.0) / 100.0,
        leader_speed / 20.0,
        rel_speed / 20.0,
        time_headway / 10.0,
        tls_dist,
        tls_red,
        tls_green,
        tls_yellow,
        lane_pos_norm,
        route_progress,
    ], dtype=np.float32)

    return state


def apply_action(veh_id, action):
    current_speed = traci.vehicle.getSpeed(veh_id)
    allowed_speed = traci.vehicle.getAllowedSpeed(veh_id)

    gap, leader_speed = get_leader_info(veh_id)
    tls_dist, tls_red, tls_green, tls_yellow = get_tls_info(veh_id)

    delta_v = ACTION_TO_DELTA_V[action]
    new_speed = current_speed + delta_v
    new_speed = max(0.0, min(new_speed, allowed_speed))

    # tls_dist normalized, max_dist=200:
    # 0.020 = 4 m
    # 0.015 = 3 m
    # 0.010 = 2 m
    # 0.0075 = 1.5 m

    leader_stopped = leader_speed < 1.0

    # Queue behavior: stop closer behind stopped car
    if leader_stopped:
        if gap > 5.0:
            new_speed = max(new_speed, 1.5)
        elif 2.5 <= gap <= 4.0:
            new_speed = min(new_speed, 0.4)
        elif gap < 2.5:
            new_speed = 0.0

    # Red light behavior: approach very close but do not cross
    if tls_red > 0.5:
        if tls_dist > 0.04 and gap > 6.0:        # farther than 8 m
            new_speed = max(new_speed, 2.0)
        elif 0.02 < tls_dist <= 0.04 and gap > 5.0:  # 4–8 m
            new_speed = max(new_speed, 0.8)
            new_speed = min(new_speed, 2.0)
        elif 0.0075 <= tls_dist <= 0.02:         # 1.5–4 m
            new_speed = min(new_speed, 0.3)
        elif tls_dist < 0.0075:                  # too close
            new_speed = 0.0

    traci.vehicle.setSpeed(veh_id, new_speed)
    return delta_v


def compute_reward(veh_id, delta_v):
    ego_speed = traci.vehicle.getSpeed(veh_id)

    gap, leader_speed = get_leader_info(veh_id)
    rel_speed = ego_speed - leader_speed

    tls_dist, tls_red, tls_green, tls_yellow = get_tls_info(veh_id)

    reward = 0.0

    # --- Time / efficiency ---
    reward -= 0.01  # time penalty

    # --- Progress (actual distance traveled since previous step) ---
    dist_now = float(traci.vehicle.getDistance(veh_id))
    dist_prev = float(_LAST_DISTANCE_BY_VEH_ID.get(veh_id, dist_now))
    delta_dist = max(0.0, dist_now - dist_prev)
    _LAST_DISTANCE_BY_VEH_ID[veh_id] = dist_now

    reward += min(0.1 * delta_dist, 0.8)  # progress

    # Safety distance
    if gap < 2.0:
        reward -= 3.0
    elif gap < 2.5:
        reward -= 1.5

    # Queue behavior shaping (small per-step values)
    if leader_speed < 1.0:
        if ego_speed < 0.5 and gap > 8.0:
            reward -= 2.0
        elif ego_speed < 0.5 and gap > 5.0:
            reward -= 1.0

        if 2.5 <= gap <= 4.0 and ego_speed < 0.7:
            reward += 0.2

    # Approaching leader too fast
    if rel_speed > 2.0 and gap < 10.0:
        reward -= 1.0

    if rel_speed > 4.0 and gap < 6.0:
        reward -= 2.0

    # Red light behavior
    if tls_red > 0.5:
        if tls_dist < 0.04 and ego_speed > 2.0:
            reward -= 2.0
        elif tls_dist < 0.10 and ego_speed > 5.0:
            reward -= 1.0

        if ego_speed < 0.5 and tls_dist > 0.04 and gap > 6.0:
            reward -= 2.0

        if 0.02 < tls_dist <= 0.08 and 0.3 <= ego_speed <= 2.5 and gap > 5.0:
            reward += 0.1

        if 0.0075 <= tls_dist <= 0.02 and ego_speed < 0.6:
            reward += 0.2

        if tls_dist < 0.0075 and ego_speed > 0.1:
            reward -= 3.0

    # Yellow caution
    if tls_yellow > 0.5 and tls_dist < 0.10 and ego_speed > 6.0:
        reward -= 1.0

    # Unnecessary stop
    red_far = tls_red > 0.5 and tls_dist > 0.04
    leader_far = gap > 8.0

    if ego_speed < 0.2 and leader_far and (tls_red < 0.5 or red_far):
        reward -= 1.0

    # Smoothness
    reward -= 0.03 * abs(delta_v)

    reward = float(np.clip(reward, -3.0, 1.0))
    return reward


def is_arrived():
    return EGO_ID in traci.simulation.getArrivedIDList()


def is_abnormal_disappearance():
    """
    Ego no longer exists and did not arrive normally.
    """
    return (not ego_exists()) and (not is_arrived())
