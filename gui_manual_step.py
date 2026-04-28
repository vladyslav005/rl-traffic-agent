"""Manual SUMO-GUI stepping helper.

Why this exists
---------------
When SUMO is started with TraCI (via `traci.start(...)`), SUMO runs in *client-controlled*
mode: the TraCI client (your Python script) is expected to advance time with
`traci.simulationStep()`. In that mode, clicking the GUI "Step" button often causes
TraCI disconnects / "simulation ended" messages because two controllers compete.

This script gives you a safe way to debug in GUI while still using TraCI:
- Starts SUMO-GUI
- Spawns the ego vehicle on a chosen route
- Advances exactly one step when *you press Enter* in the terminal

You don't need to click the GUI step button.
"""

from __future__ import annotations

import argparse

import traci

from config import EGO_ROUTE_POOL
from sumo_utils import start_sumo, spawn_ego


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route",
        default=None,
        help="Route id to spawn ego on (defaults to first route in EGO_ROUTE_POOL)",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    route_id = args.route or EGO_ROUTE_POOL[0]

    start_sumo(use_gui=True)

    ok, reason = spawn_ego(route_id)
    print(f"spawn_ego(route_id={route_id!r}) -> ok={ok}, reason={reason}")
    if not ok:
        traci.close()
        return 2

    print("\nManual stepping started.")
    print("- Press Enter to advance 1 step")
    print("- Type 'q' then Enter to quit")
    print("(Don’t use the GUI Step button while this is running.)\n")

    try:
        for step in range(args.max_steps):
            cmd = input(f"step={step}> ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break

            traci.simulationStep()

            if "ego" not in traci.vehicle.getIDList():
                print("Ego is no longer in the simulation (arrived/teleported/removed).")
                break
    finally:
        traci.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
