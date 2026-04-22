import os

import numpy as np
import traci

from Action import ACTION_TO_DELTA_V, Action
from config import *

CURRENT_USE_GUI = False


def start_sumo(use_gui=False,
               log_dir="sim_logs",
               quiet_console=False,
               hide_warnings=False):
    global CURRENT_USE_GUI
    CURRENT_USE_GUI = use_gui

    os.makedirs(log_dir, exist_ok=True)

    binary = "sumo-gui" if use_gui else SUMO_BINARY
    cmd = [
        binary,
        "-c", SUMO_CONFIG,
        "--collision.action", "warn",
        "--collision.check-junctions", "true",
        "--collision.mingap-factor", "0",
        "--aggregate-warnings", "5",
        "--log", os.path.join(log_dir, "sumo.log"),
        "--log.timestamps", "true",
    ]

    if quiet_console:
        cmd += ["--no-step-log", "true"]

    if hide_warnings:
        cmd += ["--no-warnings", "true"]

    traci.start(cmd)

def reset_sumo():
    """
    Simplest reset strategy:
    close current simulation and restart.
    Slow but clean and easy to debug.
    """
    traci.close()
    start_sumo()


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
    except traci.exceptions.TraCIException as e:
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
                    traci.vehicle.setColor(EGO_ID, (255, 0, 0, 255))
            except Exception:
                pass

            # Only works in sumo-gui; harmless if unavailable
            try:
                if CURRENT_USE_GUI:
                    traci.gui.trackVehicle("View #0", EGO_ID)
            except Exception:
                pass

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
    """
    Example traffic-aware normalized state.
    """
    ego_speed = traci.vehicle.getSpeed(veh_id)
    gap, leader_speed = get_leader_info(veh_id)
    rel_speed = ego_speed - leader_speed
    tls_dist, tls_red, tls_green, tls_yellow = get_tls_info(veh_id)

    state = np.array([
        ego_speed / 20.0,
        gap / 100.0,
        leader_speed / 20.0,
        rel_speed / 20.0,
        tls_dist,
        tls_red,
        tls_green,
        tls_yellow,
    ], dtype=np.float32)

    return state


def apply_action(veh_id, action: Action):
    current_speed = traci.vehicle.getSpeed(veh_id)
    delta_v = ACTION_TO_DELTA_V[action]
    new_speed = max(0.0, current_speed + delta_v)
    traci.vehicle.setSpeed(veh_id, new_speed)
    return delta_v


def compute_reward(veh_id, delta_v):
    """
    Safe-progress reward.
    You will tune this later.
    """
    ego_speed = traci.vehicle.getSpeed(veh_id)
    gap, leader_speed = get_leader_info(veh_id)
    rel_speed = ego_speed - leader_speed
    tls_dist, tls_red, tls_green, tls_yellow = get_tls_info(veh_id)

    reward = 0.0

    # progress reward
    reward += 0.1 * ego_speed

    # too close to leader
    if gap < 5:
        reward -= 10.0
    elif gap < 10:
        reward -= 3.0

    # closing in too fast
    if rel_speed > 2.0 and gap < 15:
        reward -= 4.0

    # approaching red light too fast
    if tls_red > 0.5 and tls_dist < 0.25 and ego_speed > 5.0:
        reward -= 5.0

    # comfort / smoothness penalty
    reward -= 0.2 * abs(delta_v)

    return reward


def is_arrived():
    return EGO_ID in traci.simulation.getArrivedIDList()


def is_abnormal_disappearance():
    """
    Ego no longer exists and did not arrive normally.
    """
    return (not ego_exists()) and (not is_arrived())

