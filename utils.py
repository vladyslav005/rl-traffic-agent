import random
from enum import IntEnum

import torch
import torch.optim as optim
import traci

from DQN import DQN
from ReplayBuffer import ReplayBuffer


class Action(IntEnum):
    SLOWER = 0
    KEEP = 1
    FASTER = 2


def get_numeric_state(car_id: str) -> list[float]:
    ego_speed = traci.vehicle.getSpeed(car_id)

    leader = traci.vehicle.getLeader(car_id)
    if leader is None:
        gap = 100.0
        leader_speed = 0.0
    else:
        leader_id, gap = leader
        leader_speed = traci.vehicle.getSpeed(leader_id)

    # light normalization helps training (no)
    return [
        ego_speed / 20.0,
        min(gap, 100.0) / 100.0,
        leader_speed / 20.0,
    ]


def apply_action(car_id: str, action: Action, delta_v: float = 2.0) -> None:
    current_speed = traci.vehicle.getSpeed(car_id)

    if action == Action.SLOWER:
        new_speed = max(0.0, current_speed - delta_v)
    elif action == Action.KEEP:
        new_speed = current_speed
    elif action == Action.FASTER:
        new_speed = current_speed + delta_v
    else:
        raise ValueError(f"Unknown action: {action}")

    traci.vehicle.setSpeed(car_id, new_speed)


def get_reward(car_id: str) -> float:
    ego_speed = traci.vehicle.getSpeed(car_id)

    leader = traci.vehicle.getLeader(car_id)
    if leader is None:
        gap = 100.0
    else:
        _, gap = leader

    reward = 0.1 * ego_speed

    if gap < 5.0:
        reward -= 10.0

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