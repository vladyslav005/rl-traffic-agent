import random

from torch import nn

from sumo_utils import *


def epsilon_by_step(global_step):
    frac = min(1.0, global_step / EPS_DECAY)
    return EPS_START + frac * (EPS_END - EPS_START)


def select_action(policy_net, state, epsilon):
    if random.random() < epsilon:
        return random.choice(list(Action))

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = policy_net(state_t)
        action_idx = int(torch.argmax(q_values, dim=1).item())
        return Action(action_idx)


def train_step(policy_net, target_net, optimizer, replay_buffer):
    if len(replay_buffer) < MIN_REPLAY_SIZE or len(replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    q_values = policy_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        next_q_values = target_net(next_states_t).max(dim=1, keepdim=True)[0]
        targets = rewards_t + GAMMA * (1.0 - dones_t) * next_q_values

    loss = nn.MSELoss()(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def run_episode(policy_net, target_net, optimizer, replay_buffer, global_step):
    route_id = random.choice(EGO_ROUTE_POOL)

    reset_sumo()
    spawn_ego(route_id)

    ################### Optional: set camera to follow ego for better visualization ###################

    # let SUMO advance one step so vehicle is fully inserted
    traci.simulationStep()

    if not ego_exists():
        return 0.0, 0, global_step, "spawn_failed"

    # try:
    #     traci.gui.trackVehicle("View #0", EGO_ID)
    #     traci.gui.setZoom("View #0", 1000)
    # except:
    #     print("Warning: GUI tracking failed, continuing without it.")

    episode_reward = 0.0
    episode_loss_values = []

    state = get_state(EGO_ID)

    # traci.gui.trackVehicle("View #0", "ego")

    for step in range(MAX_STEPS_PER_EPISODE):
        epsilon = epsilon_by_step(global_step)
        action = select_action(policy_net, state, epsilon)

        delta_v = apply_action(EGO_ID, action)

        traci.simulationStep()
        global_step += 1

        # terminal checks after the environment step
        if is_arrived():
            reward = 20.0
            next_state = np.zeros_like(state, dtype=np.float32)
            done = 1.0

            replay_buffer.push(state, int(action), reward, next_state, done)
            loss = train_step(policy_net, target_net, optimizer, replay_buffer)
            if loss is not None:
                episode_loss_values.append(loss)

            episode_reward += reward
            return episode_reward, step + 1, global_step, "arrived"

        if is_abnormal_disappearance():
            reward = -20.0
            next_state = np.zeros_like(state, dtype=np.float32)
            done = 1.0

            replay_buffer.push(state, int(action), reward, next_state, done)
            loss = train_step(policy_net, target_net, optimizer, replay_buffer)
            if loss is not None:
                episode_loss_values.append(loss)

            episode_reward += reward
            return episode_reward, step + 1, global_step, "abnormal_end"

        # normal transition
        next_state = get_state(EGO_ID)
        reward = compute_reward(EGO_ID, delta_v)
        done = 0.0

        replay_buffer.push(state, int(action), reward, next_state, done)

        loss = train_step(policy_net, target_net, optimizer, replay_buffer)
        if loss is not None:
            episode_loss_values.append(loss)

        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_reward += reward
        state = next_state

    # timeout
    if ego_exists():
        reward = -5.0
        next_state = np.zeros_like(state, dtype=np.float32)
        done = 1.0
        replay_buffer.push(state, int(Action.KEEP), reward, next_state, done)

    return episode_reward, MAX_STEPS_PER_EPISODE, global_step, "timeout"
