# %%
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

start_sumo()

state_dim = 8
action_dim = len(Action)

policy_net = DQN(state_dim, action_dim).to(DEVICE)
target_net = DQN(state_dim, action_dim).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

global_step = 0

# Log file for this run
episode_logger = EpisodeLogger(default_episode_log_path())
print(f"Logging episodes to: {episode_logger.path}")

for episode in range(NUM_EPISODES):

    ep_reward, ep_steps, global_step, end_reason = run_episode(
        policy_net, target_net, optimizer, replay_buffer, global_step
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
        total_ego_crashes=config.TOTAL_EGO_CRASHES,
        total_collision_events=config.TOTAL_COLLISION_EVENTS,
        total_ego_collisions=config.TOTAL_EGO_COLLISIONS,
        total_ego_teleports=config.TOTAL_EGO_TELEPORTS,
        total_ego_emergency_stops=config.TOTAL_EGO_EMERGENCY_STOPS,
    )

    # optional checkpoint
    if (episode + 1) % 50 == 0:
        torch.save(policy_net.state_dict(), f"dqn_training/dqn_ego_episode_{episode+1}.pth")

traci.close()
