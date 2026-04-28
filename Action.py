import os
import sys
import random
from enum import IntEnum
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Action(IntEnum):
    STRONG_BRAKE = 0
    SLOWER = 1
    KEEP = 2
    FASTER = 3
    STRONG_FASTER = 4


ACTION_TO_DELTA_V = {
    Action.STRONG_BRAKE: -2.0,
    Action.SLOWER: -1.0,
    Action.KEEP: 0.0,
    Action.FASTER: 1.0,
    Action.STRONG_FASTER: 2.0,
}
