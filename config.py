from pathlib import Path

import torch

SUMO_BINARY = "sumo"   # use "sumo" for faster training without GUI
BASE_DIR = Path(__file__).resolve().parent

SUMO_CONFIG = str(BASE_DIR / "scenarios" / "bologna_joined" / "run.sumocfg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 1500

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
    "ego_route_1",
    "ego_route_2",
    "ego_route_3",
    "ego_route_4",
    "ego_route_5",
    "ego_route_6",
    "ego_route_7",
    "ego_route_8",
    "ego_route_9",
    "ego_route_10",
    "ego_route_11",
    "ego_route_12",
    "ego_route_13",
    "ego_route_14",
    "ego_route_15",
    "ego_route_16",
    "ego_route_17",
    "ego_route_18",
    "ego_route_19",
    "ego_route_20",
    "ego_route_22",
    "ego_route_23",
    "ego_route_24",
    "ego_route_25",
    "ego_route_26",
    "ego_route_27",
    "ego_route_28",
    "ego_route_29",
    "ego_route_30",
    "ego_route_31",
    "ego_route_32",
    "ego_route_33",
    "ego_route_34",
    "ego_route_36",
    "ego_route_37",
    "ego_route_38",
    "ego_route_40",
    "ego_route_41",
    "ego_route_42",
    "ego_route_43",
    "ego_route_44",
    "ego_route_45",
    "ego_route_46",
]


VALIDATION_ROUTES = [
    "val_route_1",
    "val_route_2",
    "val_route_3",
    "val_route_4",
    "val_route_5",
    "val_route_6",
    "val_route_7",
    "val_route_8",
    "val_route_9",
    "val_route_10",
    "val_route_11",
    "val_route_12",
    "val_route_13",
    "val_route_14",
    "val_route_15",
    "val_route_16",
    "val_route_17",
    "val_route_18",
    "val_route_19",
    "val_route_20",
]

# vehicle type should exist in your SUMO files, or use "passenger"
EGO_TYPE_ID = "egoType"

TOTAL_EGO_CRASHES = 0
TOTAL_COLLISION_EVENTS = 0
TOTAL_EGO_COLLISIONS = 0
TOTAL_EGO_TELEPORTS = 0
TOTAL_EGO_EMERGENCY_STOPS = 0