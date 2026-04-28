"""
watch_one_ppo_episode_slow.py

Запуск одного PPO episode через SUMO GUI с паузой перед стартом и удержанием окна после завершения.

Запуск:
    python watch_one_ppo_episode_slow.py

Если SUMO config не нашёлся автоматически:
    python watch_one_ppo_episode_slow.py --sumo-config path/to/file.sumocfg

Если sumo-gui не находится:
    python watch_one_ppo_episode_slow.py --sumo-binary /path/to/sumo-gui

Важно:
    Этот файл специально НЕ использует sumo_utils.start_sumo(), потому что в твоём
    проекте он сейчас запускает обычный `sumo`, а для просмотра нужен `sumo-gui`.
"""

import argparse
import glob
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
    reset_sumo,
    spawn_ego,
)


# ====== Настройки под твои checkpoints ======
OBS_SIZE = 8
N_ACTIONS = len(Action)
ACTIONS = list(Action)

DEVICE = torch.device(getattr(config, "DEVICE", "cpu"))
EGO_ID = getattr(config, "EGO_ID", "ego")
EGO_ROUTE_POOL = getattr(config, "EGO_ROUTE_POOL", [])
MAX_STEPS_PER_EPISODE = getattr(config, "MAX_STEPS_PER_EPISODE", 1000)


# ====== Actor должен совпадать с архитектурой сохранённых checkpoints ======
# Твои checkpoints сохранены под:
# Actor: 8 -> 64 -> 64 -> 5
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


def get_episode_number(path):
    """Достаёт номер episode из имени файла, например ppo_ego_episode_500.pth -> 500."""
    match = re.search(r"episode_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def find_checkpoints(pattern):
    """Находит checkpoints и сортирует их по номеру episode."""
    return [Path(p) for p in sorted(glob.glob(pattern), key=get_episode_number)]


def load_actor_checkpoint(checkpoint_path):
    """Загружает только Actor. Critic и optimizer для просмотра не нужны."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    actor = Actor(OBS_SIZE, N_ACTIONS).to(DEVICE)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    return actor, checkpoint


@torch.no_grad()
def select_action_for_view(actor, state, deterministic=True):
    """
    deterministic=True  -> берём самое вероятное действие через argmax.
    deterministic=False -> выбираем случайно по вероятностям policy.
    """
    state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits = actor(state_t)

    if deterministic:
        action_t = torch.argmax(logits, dim=-1)
    else:
        dist = Categorical(logits=logits)
        action_t = dist.sample()

    return int(action_t.item())


def resolve_sumo_binary(cli_binary=None):
    """Ищет sumo-gui. Можно передать путь через --sumo-binary."""
    if cli_binary:
        return cli_binary

    # Если хочешь, можешь задать переменную окружения:
    # export SUMO_BINARY_GUI=/opt/homebrew/bin/sumo-gui
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

    # Последняя попытка: пусть ОС сама найдёт binary.
    return "sumo-gui"


def resolve_sumo_config(cli_config=None):
    """Ищет .sumocfg в config.py или в текущей папке."""
    if cli_config:
        return cli_config

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

    sumocfg_files = sorted(Path(".").glob("*.sumocfg"))
    if len(sumocfg_files) == 1:
        return str(sumocfg_files[0])

    if len(sumocfg_files) > 1:
        raise RuntimeError(
            "Нашёл несколько .sumocfg файлов. Укажи нужный явно: "
            "python watch_one_ppo_episode_slow.py --sumo-config path/to/file.sumocfg\n"
            f"Файлы: {[str(p) for p in sumocfg_files]}"
        )

    raise RuntimeError(
        "Не смог найти SUMO config. Добавь в config.py переменную SUMO_CONFIG "
        "или запусти так: python watch_one_ppo_episode_slow.py --sumo-config path/to/file.sumocfg"
    )


def start_sumo_gui(sumo_binary=None, sumo_config=None, delay_ms=50):
    """
    Принудительно запускает именно SUMO GUI.
    Не использует sumo_utils.start_sumo(), потому что там у тебя запускается command line sumo.
    """
    binary = resolve_sumo_binary(sumo_binary)
    cfg = resolve_sumo_config(sumo_config)

    cmd = [
        binary,
        "-c",
        cfg,
        "--start",
        "--delay",
        str(delay_ms),
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


def try_focus_gui():
    """Пытается привязать камеру SUMO GUI к ego car."""
    try:
        traci.gui.trackVehicle("View #0", EGO_ID)
        traci.gui.setZoom("View #0", 1500)
    except Exception as exc:
        print(f"GUI focus skipped: {exc}")


def ego_has_collision():
    """Проверяет, попал ли ego в столкновение."""
    try:
        return EGO_ID in traci.simulation.getCollidingVehiclesIDList()
    except Exception:
        return False


def watch_checkpoint(checkpoint_path, route_id, sleep_s=0.03, deterministic=True, after_spawn_wait=2.0):
    """Запускает один episode в SUMO GUI с Actor из одного checkpoint."""
    actor, checkpoint = load_actor_checkpoint(checkpoint_path)
    saved_episode = checkpoint.get("episode", get_episode_number(checkpoint_path))

    print("\n" + "=" * 80)
    print(f"Watching checkpoint: {checkpoint_path}")
    print(f"Saved episode: {saved_episode}")
    print(f"Route: {route_id}")
    print("=" * 80)

    reset_sumo()
    spawn_ego(route_id)

    # Один шаг нужен, чтобы SUMO реально вставил ego vehicle в симуляцию.
    traci.simulationStep()
    time.sleep(sleep_s)

    if not ego_exists():
        print("Ego spawn failed")
        return {
            "episode": saved_episode,
            "reward": 0.0,
            "steps": 0,
            "end_reason": "spawn_failed",
        }

    try_focus_gui()

    if after_spawn_wait > 0:
        print(f"Waiting {after_spawn_wait:.1f}s after spawn so the GUI can render the scene...")
        time.sleep(after_spawn_wait)

    total_reward = 0.0
    steps_taken = 0
    end_reason = "timeout"
    state = get_state(EGO_ID)

    for step in range(MAX_STEPS_PER_EPISODE):
        steps_taken = step + 1

        action_idx = select_action_for_view(
            actor,
            state,
            deterministic=deterministic,
        )

        # Безопаснее, чем Action(action_idx), потому что не зависит от enum values.
        action = ACTIONS[action_idx]
        delta_v = apply_action(EGO_ID, action)

        traci.simulationStep()
        time.sleep(sleep_s)

        if ego_has_collision():
            reward = -30.0
            total_reward += reward
            end_reason = "ego_crash"
            break

        if is_arrived():
            reward = 20.0
            total_reward += reward
            end_reason = "arrived"
            break

        if is_abnormal_disappearance():
            reward = -20.0
            total_reward += reward
            end_reason = "abnormal_end"
            break

        if not ego_exists():
            end_reason = "ego_disappeared"
            break

        reward = compute_reward(EGO_ID, delta_v)
        total_reward += reward
        state = get_state(EGO_ID)

    print(
        f"Result | checkpoint_episode={saved_episode} | "
        f"reward={total_reward:.2f} | "
        f"steps={steps_taken} | "
        f"end={end_reason}"
    )

    return {
        "episode": saved_episode,
        "reward": total_reward,
        "steps": steps_taken,
        "end_reason": end_reason,
    }


def choose_route(route_index=None):
    """Выбирает один route, чтобы все checkpoints сравнивались честно."""
    if not EGO_ROUTE_POOL:
        raise RuntimeError("EGO_ROUTE_POOL пустой или не найден в config.py")

    if route_index is None:
        return random.choice(EGO_ROUTE_POOL)

    if route_index < 0 or route_index >= len(EGO_ROUTE_POOL):
        raise IndexError(
            f"route_index={route_index} вне диапазона. "
            f"Доступно routes: 0..{len(EGO_ROUTE_POOL) - 1}"
        )

    return EGO_ROUTE_POOL[route_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ppo_ego_episode_500.pth",
        help="Один checkpoint для просмотра. Например: ppo_ego_episode_500.pth",
    )
    parser.add_argument(
        "--route-index",
        type=int,
        default=0,
        help="Индекс route из EGO_ROUTE_POOL. По умолчанию 0.",
    )
    parser.add_argument(
        "--route-id",
        type=str,
        default=None,
        help="Можно указать route id напрямую, например: ego_route_2. Если указано, route-index игнорируется.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Пауза между simulationStep, чтобы было видно движение. Например 0.2 или 0.5.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Если указать, action будет sampling из policy. По умолчанию используется argmax.",
    )
    parser.add_argument(
        "--sumo-binary",
        type=str,
        default=None,
        help="Путь к sumo-gui. Например: /opt/homebrew/bin/sumo-gui",
    )
    parser.add_argument(
        "--sumo-config",
        type=str,
        default=None,
        help="Путь к .sumocfg файлу, если он не указан в config.py.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=300,
        help="SUMO GUI delay в миллисекундах. Больше = медленнее в самом GUI.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=3.0,
        help="Сколько секунд ждать после открытия SUMO GUI перед подготовкой episode.",
    )
    parser.add_argument(
        "--after-spawn-wait",
        type=float,
        default=2.0,
        help="Сколько секунд ждать после появления ego car перед началом движения.",
    )
    parser.add_argument(
        "--no-wait-start",
        action="store_true",
        help="Не ждать Enter перед стартом episode.",
    )
    parser.add_argument(
        "--no-hold-open",
        action="store_true",
        help="Не удерживать окно SUMO открытым после завершения episode.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint не найден: {checkpoint_path}")
        return

    if args.route_id is not None:
        route_id = args.route_id
    else:
        route_id = choose_route(args.route_index)

    deterministic = not args.stochastic

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Route: {route_id}")
    print(f"Deterministic mode: {deterministic}")
    print(f"Sleep per step: {args.sleep}")

    start_sumo_gui(
        sumo_binary=args.sumo_binary,
        sumo_config=args.sumo_config,
        delay_ms=args.delay_ms,
    )

    if args.startup_wait > 0:
        print(f"Waiting {args.startup_wait:.1f}s so SUMO GUI can fully open...")
        time.sleep(args.startup_wait)

    if not args.no_wait_start:
        input("SUMO GUI opened. Press Enter here to START the episode...")

    try:
        result = watch_checkpoint(
            checkpoint_path=checkpoint_path,
            route_id=route_id,
            sleep_s=args.sleep,
            deterministic=deterministic,
            after_spawn_wait=args.after_spawn_wait,
        )
    finally:
        if not args.no_hold_open:
            try:
                input("Episode finished. Press Enter here to CLOSE SUMO GUI...")
            except EOFError:
                time.sleep(10)
        try:
            traci.close()
        except Exception:
            pass

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
