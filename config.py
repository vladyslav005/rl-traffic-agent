import torch

SUMO_BINARY = "sumo"   # use "sumo" for faster training without GUI
SUMO_CONFIG = "scenarios/bologna_joined/run.sumocfg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 500

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 50000
MIN_REPLAY_SIZE = 2000
TARGET_UPDATE_FREQ = 500

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 30000   # decay over global steps

EGO_ID = "ego"

# predefined route IDs must exist in your SUMO route file
EGO_ROUTE_POOL = [
    # "ego_route_1",
    "ego_route_2",
    # "ego_route_3",
    "ego_route_4",
    "ego_route_5",
    "ego_route_6",
    "ego_route_7",
    "ego_route_8",
]
# vehicle type should exist in your SUMO files, or use "passenger"
EGO_TYPE_ID = "egoType"

TOTAL_EGO_CRASHES = 0
TOTAL_COLLISION_EVENTS = 0
TOTAL_EGO_COLLISIONS = 0
TOTAL_EGO_TELEPORTS = 0
TOTAL_EGO_EMERGENCY_STOPS = 0