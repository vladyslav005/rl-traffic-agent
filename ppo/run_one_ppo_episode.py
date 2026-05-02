"""
run_one_ppo_episode.py

Один файл для запуска просмотра одного PPO episode в SUMO GUI.

Как запускать из папки проекта:
    python run_one_ppo_episode.py

Что можно поменять быстро:
    1) CHECKPOINT_PATH
    2) ROUTE_ID
    3) SLEEP_PER_STEP
    4) SUMO_GUI_DELAY_MS

Важно:
    - Файл специально НЕ вызывает reset_sumo().
    - SUMO запускается через sumo-gui, а не через обычный sumo.
    - Actor внутри файла совпадает с твоими checkpoints: 8 -> 64 -> 64 -> 5.
"""

import os
import random
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

import traci
import config
from Action import Action
from sumo_utils import (
    apply_action,
    compute_reward,
    ego_exists,
    get_state,
    is_abnormal_disappearance,
    is_arrived,
    spawn_ego,
)


# =============================================================================
# НАСТРОЙКИ ЗАПУСКА
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)
CHECKPOINT_PATH = str(BASE_DIR / "ppo_training" / "ppo_ego_episode_50.pth")

# Можно указать конкретный маршрут, например "ego_route_2".
# Если поставить None, маршрут выберется из EGO_ROUTE_POOL по ROUTE_INDEX.
ROUTE_ID = "ego_route_2"
ROUTE_INDEX = 0

# Чем больше, тем медленнее движение между шагами Python/TraCI.
SLEEP_PER_STEP = 0.5

# Чем больше, тем медленнее проигрывание внутри SUMO GUI.
SUMO_GUI_DELAY_MS = 100

# Паузы, чтобы окно успевало открыться и ты мог посмотреть.
STARTUP_WAIT_SECONDS = 3.0
AFTER_SPAWN_WAIT_SECONDS = 2.0

# True  -> агент выбирает самое вероятное действие через argmax.
# False -> агент выбирает действие случайно по вероятностям policy.
DETERMINISTIC = True

# Если sumo-gui не находится автоматически, укажи путь руками, например:
# SUMO_BINARY = "/opt/homebrew/bin/sumo-gui"
SUMO_BINARY = None

# Если .sumocfg не находится автоматически, укажи путь руками, например:
# SUMO_CONFIG = "osm.sumocfg"
SUMO_CONFIG = None


# =============================================================================
# ПАРАМЕТРЫ ENV / MODEL
# =============================================================================

OBS_SIZE = 12
N_ACTIONS = len(Action)
ACTIONS = list(Action)

DEVICE = torch.device(getattr(config, "DEVICE", "cpu"))
EGO_ID = getattr(config, "EGO_ID", "ego")
EGO_ROUTE_POOL = getattr(config, "EGO_ROUTE_POOL", [])
MAX_STEPS_PER_EPISODE = getattr(config, "MAX_STEPS_PER_EPISODE", 1000)


# =============================================================================
# ACTOR: должен совпадать с тем, как были сохранены checkpoints
# Actor: 8 -> 64 -> 64 -> 5
# =============================================================================

class Actor(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# HELPERS
# =============================================================================

def get_episode_number(path):
    match = re.search(r"episode_(\d+)", str(path))
    return int(match.group(1)) if match else "unknown"


def resolve_sumo_binary():
    if SUMO_BINARY:
        return SUMO_BINARY

    env_binary = os.environ.get("SUMO_BINARY_GUI")
    if env_binary:
        return env_binary

    from_path = shutil.which("sumo-gui")
    if from_path:
        return from_path

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo-gui"
        if candidate.exists():
            return str(candidate)

    return "sumo-gui"


def resolve_sumo_config():
    if SUMO_CONFIG:
        return SUMO_CONFIG

    possible_config_attrs = [
        "SUMO_CONFIG",
        "SUMO_CFG",
        "SUMOCFG_FILE",
        "SUMO_CONFIG_FILE",
        "CFG_FILE",
        "SUMO_CFG_FILE",
    ]

    for attr in possible_config_attrs:
        value = getattr(config, attr, None)
        if value:
            return str(value)

    sumocfg_files = sorted(Path("..").glob("*.sumocfg"))
    if len(sumocfg_files) == 1:
        return str(sumocfg_files[0])

    if len(sumocfg_files) > 1:
        raise RuntimeError(
            "Нашёл несколько .sumocfg файлов. Укажи нужный в SUMO_CONFIG сверху файла.\n"
            f"Файлы: {[str(p) for p in sumocfg_files]}"
        )

    raise RuntimeError(
        "Не смог найти .sumocfg. Укажи путь в SUMO_CONFIG сверху файла."
    )


def choose_route():
    if ROUTE_ID is not None:
        return ROUTE_ID

    if not EGO_ROUTE_POOL:
        raise RuntimeError("EGO_ROUTE_POOL пустой или не найден в config.py")

    if ROUTE_INDEX < 0 or ROUTE_INDEX >= len(EGO_ROUTE_POOL):
        raise IndexError(
            f"ROUTE_INDEX={ROUTE_INDEX} вне диапазона. "
            f"Доступно: 0..{len(EGO_ROUTE_POOL) - 1}"
        )

    return EGO_ROUTE_POOL[ROUTE_INDEX]


def start_sumo_gui():
    binary = resolve_sumo_binary()
    cfg = resolve_sumo_config()

    cmd = [
        binary,
        "-c",
        cfg,
        "--start",
        "--delay",
        str(SUMO_GUI_DELAY_MS),
        "--quit-on-end",
        "false",
    ]

    print("\nStarting SUMO GUI:")
    print(" ".join(cmd))

    try:
        traci.close(False)
    except Exception:
        pass

    traci.start(cmd)


def load_actor(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    actor = Actor(OBS_SIZE, N_ACTIONS).to(DEVICE)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    return actor, checkpoint


@torch.no_grad()
def select_action(actor, state):
    state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits = actor(state_t)

    if DETERMINISTIC:
        action_t = torch.argmax(logits, dim=-1)
    else:
        dist = Categorical(logits=logits)
        action_t = dist.sample()

    return int(action_t.item())


def try_focus_gui():
    try:
        traci.gui.trackVehicle("View #0", EGO_ID)
        traci.gui.setZoom("View #0", 1500)
    except Exception as exc:
        print(f"GUI focus skipped: {exc}")


def ego_has_collision():
    try:
        return EGO_ID in traci.simulation.getCollidingVehiclesIDList()
    except Exception:
        return False


# =============================================================================
# MAIN EPISODE
# =============================================================================

def run_episode():
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")

    route_id = choose_route()
    actor, checkpoint = load_actor(checkpoint_path)
    saved_episode = checkpoint.get("episode", get_episode_number(checkpoint_path))

    print("\n" + "=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved episode: {saved_episode}")
    print(f"Route: {route_id}")
    print(f"Deterministic: {DETERMINISTIC}")
    print(f"Sleep per step: {SLEEP_PER_STEP}")
    print("=" * 80)

    start_sumo_gui()

    if STARTUP_WAIT_SECONDS > 0:
        print(f"Waiting {STARTUP_WAIT_SECONDS:.1f}s so SUMO GUI can fully open...")
        time.sleep(STARTUP_WAIT_SECONDS)

    input("SUMO GUI opened. Press Enter here to START the episode...")

    # ВАЖНО: reset_sumo() не вызываем.
    spawn_ego(route_id)
    traci.simulationStep()
    time.sleep(SLEEP_PER_STEP)

    if not ego_exists():
        return {
            "episode": saved_episode,
            "reward": 0.0,
            "steps": 0,
            "end_reason": "spawn_failed",
        }

    try_focus_gui()

    if AFTER_SPAWN_WAIT_SECONDS > 0:
        print(f"Waiting {AFTER_SPAWN_WAIT_SECONDS:.1f}s after spawn...")
        time.sleep(AFTER_SPAWN_WAIT_SECONDS)

    total_reward = 0.0
    steps_taken = 0
    end_reason = "timeout"
    state = get_state(EGO_ID)

    for step in range(MAX_STEPS_PER_EPISODE):
        steps_taken = step + 1

        action_idx = select_action(actor, state)
        action = ACTIONS[action_idx]

        delta_v = apply_action(EGO_ID, action)
        traci.simulationStep()
        time.sleep(SLEEP_PER_STEP)

        if ego_has_collision():
            total_reward += -30.0
            end_reason = "ego_crash"
            break

        if is_arrived():
            total_reward += 20.0
            end_reason = "arrived"
            break

        if is_abnormal_disappearance():
            total_reward += -20.0
            end_reason = "abnormal_end"
            break

        if not ego_exists():
            end_reason = "ego_disappeared"
            break

        reward = compute_reward(EGO_ID, delta_v)
        total_reward += reward
        state = get_state(EGO_ID)

    return {
        "episode": saved_episode,
        "reward": total_reward,
        "steps": steps_taken,
        "end_reason": end_reason,
    }


def main():
    result = None

    try:
        result = run_episode()
    finally:
        try:
            input("Episode finished. Press Enter here to CLOSE SUMO GUI...")
        except EOFError:
            time.sleep(10)

        try:
            traci.close()
        except Exception:
            pass

    if result is not None:
        print("\n" + "=" * 80)
        print("ONE EPISODE RESULT")
        print("=" * 80)
        print(
            f"episode={result['episode']} | "
            f"reward={result['reward']:.2f} | "
            f"steps={result['steps']} | "
            f"end={result['end_reason']}"
        )


if __name__ == "__main__":
    main()
