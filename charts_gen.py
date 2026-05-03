import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Input files
TRAIN_PPO = "train_PPO.tsv"
TRAIN_DQN = "train_DQN.tsv"
VAL_PPO = "validation_results_PPO.tsv"
VAL_DQN = "validation_results_DQN.tsv"

# Output folder
OUT_DIR = Path(".")
OUT_DIR.mkdir(exist_ok=True)


def load_train(path, algorithm_name):
    df = pd.read_csv(path, sep="\t")
    df["algorithm"] = algorithm_name
    return df


def load_validation(path, algorithm_name):
    df = pd.read_csv(path, sep="\t")
    df["algorithm"] = algorithm_name
    return df


# =========================
# Load data
# =========================

train_ppo = load_train(TRAIN_PPO, "PPO")
train_dqn = load_train(TRAIN_DQN, "DQN")

val_ppo = load_validation(VAL_PPO, "PPO")
val_dqn = load_validation(VAL_DQN, "DQN")

train_df = pd.concat([train_ppo, train_dqn], ignore_index=True)
val_df = pd.concat([val_ppo, val_dqn], ignore_index=True)


# =========================
# 1. Training reward rolling average
# =========================

plt.figure(figsize=(10, 5))

for algorithm, group in train_df.groupby("algorithm"):
    group = group.sort_values("episode")
    rolling_reward = group["reward"].rolling(window=25, min_periods=1).mean()
    plt.plot(group["episode"], rolling_reward, label=algorithm)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training reward rolling average")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_DIR / "training_reward_rolling.png", dpi=300)
plt.close()


# =========================
# 2. Validation reward by traffic scale
# =========================

reward_by_scale = (
    val_df
    .groupby(["algorithm", "traffic_scale"])["reward"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

for algorithm, group in reward_by_scale.groupby("algorithm"):
    group = group.sort_values("traffic_scale")
    plt.plot(
        group["traffic_scale"],
        group["reward"],
        marker="o",
        label=algorithm
    )

plt.xlabel("Traffic scale")
plt.ylabel("Average reward")
plt.title("Validation reward by traffic scale")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_DIR / "validation_reward_by_scale.png", dpi=300)
plt.close()


# =========================
# 3. Validation steps by traffic scale
# =========================

steps_by_scale = (
    val_df
    .groupby(["algorithm", "traffic_scale"])["steps"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

for algorithm, group in steps_by_scale.groupby("algorithm"):
    group = group.sort_values("traffic_scale")
    plt.plot(
        group["traffic_scale"],
        group["steps"],
        marker="o",
        label=algorithm
    )

plt.xlabel("Traffic scale")
plt.ylabel("Average number of steps")
plt.title("Validation steps by traffic scale")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_DIR / "validation_steps_by_scale.png", dpi=300)
plt.close()


print("Charts generated:")
print(" - training_reward_rolling.png")
print(" - validation_reward_by_scale.png")
print(" - validation_steps_by_scale.png")