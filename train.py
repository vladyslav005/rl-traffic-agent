# %%
import random

import traci
import config
from utils import epsilon_by_step, run_episode
from ReplayBuffer import ReplayBuffer
from torch import optim
from config import *
from DQN import DQN
from Action import Action
from sumo_utils import start_sumo
from logger_utils import EpisodeLogger, default_episode_log_path
import sys

# Choose whether to resume from a checkpoint.
# RESUME_PATH = "./dqn_training/dqn_ego_episode_500.pth"
# RESUME_OPTIM_PATH = "./dqn_training/dqn_ego_episode_500.optim.pth"
RESUME_PATH = False #"./dqn_training/dqn_ego_episode_500.pth"
RESUME_OPTIM_PATH = False #"./dqn_training/dqn_ego_episode_500.optim.pth"
START_EPISODE = 0

# Start SUMO once; each episode calls reset_sumo() internally.
start_sumo(use_gui=False)

state_dim = 12
action_dim = len(Action)

policy_net = DQN(state_dim, action_dim).to(DEVICE)
target_net = DQN(state_dim, action_dim).to(DEVICE)

# (1) Load weights
if RESUME_PATH:
    policy_net.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))

# (2) Target starts equal to policy
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# (3) Load optimizer state if available (recommended to truly "continue")
if RESUME_OPTIM_PATH:
    try:
        optimizer.load_state_dict(torch.load(RESUME_OPTIM_PATH, map_location=DEVICE))
    except FileNotFoundError:
        pass

replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

global_step = 0

# Log file for this run
episode_logger = EpisodeLogger(default_episode_log_path())
print(f"Logging episodes to: {episode_logger.path}")

for episode in range(START_EPISODE, START_EPISODE + NUM_EPISODES):

    route_id = random.choice(EGO_ROUTE_POOL)

    ep_reward, ep_steps, global_step, end_reason = run_episode(
        policy_net, target_net, optimizer, replay_buffer, global_step, route_id
    )


    epsilon = epsilon_by_step(global_step)

    # print(
    #     f"Episode {episode:4d} | "
    #     f"reward={ep_reward:8.2f} | "
    #     f"steps={ep_steps:4d} | "
    #     f"end={end_reason} | "
    #     f"TOTAL_EGO_CRASHES={config.TOTAL_EGO_CRASHES} | "
    #     f"TOTAL_COLLISION_EVENTS={config.TOTAL_COLLISION_EVENTS} | "
    #     f"TOTAL_EGO_COLLISIONS={config.TOTAL_EGO_COLLISIONS} | "
    #     f"TOTAL_EGO_TELEPORTS={config.TOTAL_EGO_TELEPORTS} | "
    #     f"TOTAL_EGO_EMERGENCY_STOPS={config.TOTAL_EGO_EMERGENCY_STOPS}"
    # )

    episode_logger.log(
        episode=episode,
        reward=ep_reward,
        steps=ep_steps,
        end_reason=end_reason,
        route_id=route_id,
        total_ego_crashes=config.TOTAL_EGO_CRASHES,
        total_collision_events=config.TOTAL_COLLISION_EVENTS,
        total_ego_collisions=config.TOTAL_EGO_COLLISIONS,
        total_ego_teleports=config.TOTAL_EGO_TELEPORTS,
        total_ego_emergency_stops=config.TOTAL_EGO_EMERGENCY_STOPS,
    )

    # optional checkpoint
    if (episode + 1) % 50 == 0:
        torch.save(policy_net.state_dict(), f"dqn_training_5/dqn_ego_episode_{episode+1}.pth")

traci.close()