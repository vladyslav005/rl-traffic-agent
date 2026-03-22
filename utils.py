import random
from enum import IntEnum

import torch
import torch.optim as optim
import traci

from DQN import DQN
from ReplayBuffer import ReplayBuffer


class Action(IntEnum):
    STRONG_BRAKE = 0
    SLOWER = 1
    KEEP = 2
    FASTER = 3
    STRONG_FASTER = 4


def tls_state_to_features(tls_state: str) -> tuple[float, float, float]:
    """
    Map SUMO TLS state char to 3 features:
    [is_red_like, is_yellow_like, is_green_like]
    Unknown / no-light -> all zeros
    """
    if not tls_state:
        return 0.0, 0.0, 0.0

    c = tls_state[0]

    # SUMO TLS state chars commonly include r/y/g and upper-case variants
    if c in ("r", "R"):
        return 1.0, 0.0, 0.0
    if c in ("y", "Y"):
        return 0.0, 1.0, 0.0
    if c in ("g", "G"):
        return 0.0, 0.0, 1.0

    return 0.0, 0.0, 0.0

def get_next_tls_info(car_id: str) -> tuple[float, float, float, float]:
    """
    Returns:
        tls_dist_norm, is_red, is_yellow, is_green
    """
    tls_list = traci.vehicle.getNextTLS(car_id)

    if not tls_list:
        return 1.0, 0.0, 0.0, 0.0

    # First upcoming TLS on route
    tls_id, tls_index, tls_dist, tls_state = tls_list[0]

    tls_dist_norm = min(float(tls_dist), 100.0) / 100.0
    is_red, is_yellow, is_green = tls_state_to_features(tls_state)

    return tls_dist_norm, is_red, is_yellow, is_green

def get_numeric_state(car_id: str) -> list[float]:
    ego_speed = traci.vehicle.getSpeed(car_id)

    leader = traci.vehicle.getLeader(car_id)
    if leader is None:
        gap = 100.0
        leader_speed = ego_speed
    else:
        leader_id, gap = leader
        leader_speed = traci.vehicle.getSpeed(leader_id)

    relative_speed = ego_speed - leader_speed

    tls_dist, tls_red, tls_yellow, tls_green = get_next_tls_info(car_id)

    return [
        ego_speed / 20.0,                 # 0
        min(gap, 100.0) / 100.0,         # 1
        leader_speed / 20.0,             # 2
        relative_speed / 20.0,           # 3
        tls_dist,                        # 4
        tls_red,                         # 5
        tls_green,                       # 6
        # intentionally skipping yellow in first version to keep dim smaller
    ]


def apply_action(car_id: str, action: Action) -> float:
    current_speed = traci.vehicle.getSpeed(car_id)

    if action == Action.STRONG_BRAKE:
        delta_v = -4.0
    elif action == Action.SLOWER:
        delta_v = -2.0
    elif action == Action.KEEP:
        delta_v = 0.0
    elif action == Action.FASTER:
        delta_v = 2.0
    elif action == Action.STRONG_FASTER:
        delta_v = 4.0
    else:
        raise ValueError(f"Unknown action: {action}")

    new_speed = max(0.0, current_speed + delta_v)
    traci.vehicle.setSpeed(car_id, new_speed)

    return delta_v


def get_reward(car_id: str, delta_v: float) -> float:
    ego_speed = traci.vehicle.getSpeed(car_id)

    leader = traci.vehicle.getLeader(car_id)
    if leader is None:
        gap = 100.0
        leader_speed = ego_speed
    else:
        leader_id, gap = leader
        leader_speed = traci.vehicle.getSpeed(leader_id)

    relative_speed = ego_speed - leader_speed

    tls_dist, tls_red, tls_yellow, tls_green = get_next_tls_info(car_id)

    reward = 0.0

    # 1) progress reward
    reward += 0.05 * ego_speed

    # 2) punish unsafe following distance
    if gap < 20.0:
        reward -= (20.0 - gap) * 0.2

    # 3) punish closing in on leader too fast
    if relative_speed > 0 and gap < 15.0:
        reward -= relative_speed * 0.5

    # 4) punish approaching a red light too fast
    # if red is ahead and close, high speed should be discouraged
    if tls_red > 0.5 and tls_dist < 0.30:   # within ~30 m
        reward -= ego_speed * 0.3

    # 5) small comfort penalty for jerky actions
    reward -= abs(delta_v) * 0.05

    return reward


def choose_action(policy_net: DQN, state: list[float], epsilon: float) -> Action:
    if random.random() < epsilon:
        return Action(random.randint(0, 2))

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(state_t)
        action_idx = torch.argmax(q_values, dim=1).item()
        return Action(action_idx)


def train_step(
    model: DQN,
    optimizer: optim.Optimizer,
    state: list[float],
    action: Action,
    reward: float,
    next_state: list[float],
    gamma: float = 0.99,
) -> float:
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)       # [1, 3]
    next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # [1, 3]
    reward_t = torch.tensor([reward], dtype=torch.float32)                # [1]

    q_values = model(state_t)                    # [1, 3]
    current_q = q_values[0, int(action)]         # scalar

    with torch.no_grad():
        next_q_values = model(next_state_t)      # [1, 3]
        max_next_q = torch.max(next_q_values, dim=1).values[0]  # scalar
        target_q = reward_t[0] + gamma * max_next_q

    loss = (target_q - current_q) ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def train_step_batch(
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float = 0.99,
) -> float | None:
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32)

    q_values = policy_net(states_t)
    current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states_t)
        max_next_q = torch.max(next_q_values, dim=1).values
        target_q = rewards_t + gamma * max_next_q * (1.0 - dones_t)

    loss = torch.mean((target_q - current_q) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()