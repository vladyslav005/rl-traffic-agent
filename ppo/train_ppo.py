from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

import config
import traci
from Action import Action
from ppo.PPO_actor_critic import Actor, Critic
from config import DEVICE, EGO_ID, EGO_ROUTE_POOL, GAMMA, MAX_STEPS_PER_EPISODE, NUM_EPISODES
from logger_utils import EpisodeLogger, TsvLogger, default_episode_log_path
from sumo_utils import (
    apply_action,
    compute_reward,
    ego_exists,
    get_state,
    is_abnormal_disappearance,
    is_arrived,
    reset_sumo,
    spawn_ego,
    start_sumo,
)
from traci.exceptions import FatalTraCIError

# Keep in sync with sumo_utils.get_state(), which currently returns 12 features.
OBS_SIZE = 12
N_ACTIONS = len(Action)



PPO_LR = 3e-4
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 256
PPO_CLIP_EPS = 0.2
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03

ROLLOUT_STEPS = 2048 # number of steps to collect across episodes before each PPO update; can be higher than MAX_STEPS_PER_EPISODE to allow multi-episode batches
CHECKPOINT_FREQ = 50 # save model every N episodes
SEED = 42
MAX_CONSECUTIVE_SPAWN_FAILURES = 10

NUM_EPISODES = 1000
# RESUME_CHECKPOINT = "ppo_training/ppo_ego_episode_500.pth"
RESUME_CHECKPOINT = None

@dataclass
class RolloutBuffer: # stores trajectories collected from one or more episodes, to be used for a PPO update
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: float,
        value: float,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class EpisodeResult:
    rollout: RolloutBuffer
    reward: float
    steps: int
    global_step: int
    end_reason: str
    last_value: float


def count_end_reasons(reasons: list[str], prefix: str) -> int:
    return sum(reason == prefix or reason.startswith(f"{prefix}:") for reason in reasons)


def reset_counters() -> None:
    config.TOTAL_EGO_CRASHES = 0
    config.TOTAL_COLLISION_EVENTS = 0
    config.TOTAL_EGO_COLLISIONS = 0
    config.TOTAL_EGO_TELEPORTS = 0
    config.TOTAL_EGO_EMERGENCY_STOPS = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def select_action(actor: Actor, critic: Critic, state: np.ndarray, deterministic: bool = False):
    state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits = actor(state_t)
    dist = Categorical(logits=logits)
    action_t = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
    log_prob = dist.log_prob(action_t)
    value = critic(state_t).squeeze(-1)
    return int(action_t.item()), float(log_prob.item()), float(value.item())


def compute_gae(
    rewards: list[float],
    dones: list[float],
    values: list[float],
    last_value: float,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    values_np = np.asarray(values + [last_value], dtype=np.float32)

    advantages = np.zeros_like(rewards_np, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards_np))):
        mask = 1.0 - dones_np[t]
        delta = rewards_np[t] + gamma * values_np[t + 1] * mask - values_np[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + values_np[:-1]
    return returns, advantages


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    returns_var = np.var(returns)
    if returns_var < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values) / returns_var)


def update_safety_counters() -> bool:
    colliding_ids = traci.simulation.getCollidingVehiclesIDList()
    collision_events = traci.simulation.getCollisions()
    teleport_ids = traci.simulation.getStartingTeleportIDList()
    emergency_ids = traci.simulation.getEmergencyStoppingVehiclesIDList()

    config.TOTAL_COLLISION_EVENTS += len(collision_events)

    ego_collision = EGO_ID in colliding_ids
    ego_teleport = EGO_ID in teleport_ids
    ego_emergency = EGO_ID in emergency_ids

    if ego_collision:
        config.TOTAL_EGO_COLLISIONS += 1
    if ego_teleport:
        config.TOTAL_EGO_TELEPORTS += 1
    if ego_emergency:
        config.TOTAL_EGO_EMERGENCY_STOPS += 1
    if ego_collision or ego_teleport or ego_emergency:
        config.TOTAL_EGO_CRASHES += 1

    return ego_collision


def run_ppo_episode(actor: Actor, critic: Critic, global_step: int) -> EpisodeResult:
    rollout = RolloutBuffer()
    route_id = random.choice(EGO_ROUTE_POOL)

    reset_sumo(use_gui=False)
    spawn_ok, spawn_reason = spawn_ego(route_id)

    if not spawn_ok:
        return EpisodeResult(
            rollout=rollout,
            reward=0.0,
            steps=0,
            global_step=global_step,
            end_reason=f"spawn_failed:{spawn_reason}",
            last_value=0.0,
        )

    if not ego_exists():
        return EpisodeResult(
            rollout=rollout,
            reward=0.0,
            steps=0,
            global_step=global_step,
            end_reason="spawn_failed:ego_missing_after_spawn",
            last_value=0.0,
        )

    episode_reward = 0.0
    end_reason = "timeout"
    state = get_state(EGO_ID)
    if state.shape[0] != OBS_SIZE:
        raise RuntimeError(
            f"Unexpected state size {state.shape[0]}. "
            f"train_ppo.py expects {OBS_SIZE}; keep it in sync with sumo_utils.get_state()."
        )
    steps_taken = 0
    last_value = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        action_idx, log_prob, value = select_action(actor, critic, state)
        delta_v = apply_action(EGO_ID, Action(action_idx))

        traci.simulationStep()
        global_step += 1
        steps_taken = step + 1

        ego_collision = update_safety_counters()

        if ego_collision:
            reward = -30.0
            rollout.add(state, action_idx, log_prob, reward, 1.0, value)
            episode_reward += reward
            end_reason = "ego_crash"
            break

        if is_arrived():
            reward = 20.0
            rollout.add(state, action_idx, log_prob, reward, 1.0, value)
            episode_reward += reward
            end_reason = "arrived"
            break

        if is_abnormal_disappearance():
            reward = -20.0
            rollout.add(state, action_idx, log_prob, reward, 1.0, value)
            episode_reward += reward
            end_reason = "abnormal_end"
            break

        next_state = get_state(EGO_ID)
        reward = compute_reward(EGO_ID, delta_v)
        rollout.add(state, action_idx, log_prob, reward, 0.0, value)

        episode_reward += reward
        state = next_state
    else:
        if ego_exists():
            with torch.no_grad():
                last_value = float(
                    critic(torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)).item()
                )

    return EpisodeResult(
        rollout=rollout,
        reward=episode_reward,
        steps=steps_taken,
        global_step=global_step,
        end_reason=end_reason,
        last_value=last_value,
    )

def concat_rollout(valid_episodes, attr, dtype):
    return np.concatenate(
        [np.asarray(getattr(ep.rollout, attr), dtype=dtype) for ep in valid_episodes],
        axis=0,
    )


def ppo_update(
    actor: Actor,
    critic: Critic,
    optimizer: optim.Optimizer,
    batch_episodes: list[EpisodeResult],
) -> dict[str, float]:
    valid_episodes = [episode for episode in batch_episodes if len(episode.rollout) > 0]
    if not valid_episodes:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
            "epochs_ran": 0.0,
            "samples": 0.0,
        }

    states_np = concat_rollout(valid_episodes, "states", np.float32)
    actions_np = concat_rollout(valid_episodes, "actions", np.int64)
    old_log_probs_np = concat_rollout(valid_episodes, "log_probs", np.float32)
    values_np = concat_rollout(valid_episodes, "values", np.float32)


    returns_chunks = []
    advantages_chunks = []
    for episode in valid_episodes:
        returns_np, advantages_np = compute_gae(
            episode.rollout.rewards,
            episode.rollout.dones,
            episode.rollout.values,
            episode.last_value,
        )
        returns_chunks.append(returns_np)
        advantages_chunks.append(advantages_np)

    returns_np = np.concatenate(returns_chunks, axis=0)
    advantages_np = np.concatenate(advantages_chunks, axis=0)

    states = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(actions_np, dtype=torch.long, device=DEVICE)
    old_log_probs = torch.as_tensor(old_log_probs_np, dtype=torch.float32, device=DEVICE)
    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=DEVICE)
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=DEVICE)

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
    batch_size = min(PPO_BATCH_SIZE, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy_losses = []
    value_losses = []
    entropies = []
    total_losses = []
    approx_kls = []
    clip_fractions = []
    epochs_ran = 0

    params = list(actor.parameters()) + list(critic.parameters())

    for _ in range(PPO_EPOCHS):
        epoch_approx_kls = []
        for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in loader:
            # print("batch_states", batch_states.shape)

            logits = actor(batch_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            values_pred = critic(batch_states).squeeze(-1)

            log_ratio = new_log_probs - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, batch_returns)
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            optimizer.step()

            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > PPO_CLIP_EPS).float().mean()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            total_losses.append(float(loss.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))
            epoch_approx_kls.append(float(approx_kl.item()))

        epochs_ran += 1
        if TARGET_KL is not None and epoch_approx_kls and float(np.mean(epoch_approx_kls)) > TARGET_KL:
            break

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "explained_variance": explained_variance(values_np, returns_np),
        "epochs_ran": float(epochs_ran),
        "samples": float(len(dataset)),
    }


def save_checkpoint(
    actor: Actor,
    critic: Critic,
    optimizer: optim.Optimizer,
    *,
    episode: int,
    global_step: int,
    update_idx: int,
) -> str:
    checkpoint_dir = Path(__file__).resolve().parent / "ppo_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"ppo_ego_episode_{episode}.pth"
    torch.save(
        {
            "episode": episode,
            "global_step": global_step,
            "update_idx": update_idx,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return str(checkpoint_path)

def load_checkpoint(
    actor: Actor,
    critic: Critic,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
) -> tuple[int, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    episode = int(checkpoint["episode"])
    global_step = int(checkpoint["global_step"])
    update_idx = int(checkpoint["update_idx"])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Resume from episode={episode}, global_step={global_step}, update_idx={update_idx}")

    return episode, global_step, update_idx


def main() -> None:
    if not EGO_ROUTE_POOL:
        raise RuntimeError("EGO_ROUTE_POOL is empty. Configure at least one ego route in config.py.")

    set_seed(SEED)
    reset_counters()

    actor = Actor(OBS_SIZE, N_ACTIONS).to(DEVICE)
    critic = Critic(OBS_SIZE).to(DEVICE)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=PPO_LR)



    episode_logger = EpisodeLogger(default_episode_log_path(prefix="ppo_episodes"))
    update_logger = TsvLogger(
        default_episode_log_path(prefix="ppo_updates"),
        (
            "timestamp",
            "update_idx",
            "episode_start",
            "episode_end",
            "episodes_in_update",
            "rollout_steps",
            "mean_reward",
            "mean_steps",
            "arrived",
            "ego_crash",
            "abnormal_end",
            "timeout",
            "spawn_failed",
            "policy_loss",
            "value_loss",
            "entropy",
            "loss",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
            "epochs_ran",
            "samples",
        ),
    )


    consecutive_spawn_failures = 0


    if RESUME_CHECKPOINT:
        episodes_completed, global_step, update_idx = load_checkpoint(
            actor,
            critic,
            optimizer,
            RESUME_CHECKPOINT,
        )
    else:
        global_step = 0
        update_idx = 0
        episodes_completed = 0

    print(f"Episode log: {episode_logger.path}")
    print(f"Update log: {update_logger.path}")

    try:
        start_sumo(use_gui=False)
    except (FatalTraCIError, TypeError) as exc:
        raise RuntimeError(
            "Failed to start SUMO/TraCI. If this happens inside a sandboxed environment, "
            "local socket creation may be blocked and the training loop must be run outside the sandbox."
        ) from exc
    try:
        while episodes_completed < NUM_EPISODES:
            batch_results: list[EpisodeResult] = []
            batch_steps = 0
            batch_start_episode = episodes_completed
            batch_rewards = []
            batch_episode_lengths = []
            batch_end_reasons = []

            episodes_until_checkpoint = CHECKPOINT_FREQ - (episodes_completed % CHECKPOINT_FREQ)
            batch_episode_cap = min(episodes_until_checkpoint, NUM_EPISODES - episodes_completed)

            while (
                episodes_completed < NUM_EPISODES
                and len(batch_rewards) < batch_episode_cap
                and (batch_steps < ROLLOUT_STEPS or not batch_results)
            ):
                result = run_ppo_episode(actor, critic, global_step)
                global_step = result.global_step
                episodes_completed += 1

                batch_rewards.append(result.reward)
                batch_episode_lengths.append(result.steps)
                batch_end_reasons.append(result.end_reason)

                print(
                    f"Episode {episodes_completed:4d} | "
                    f"reward={result.reward:8.2f} | "
                    f"steps={result.steps:4d} | "
                    f"end={result.end_reason} | "
                    f"TOTAL_EGO_CRASHES={config.TOTAL_EGO_CRASHES} | "
                    f"TOTAL_COLLISION_EVENTS={config.TOTAL_COLLISION_EVENTS} | "
                    f"TOTAL_EGO_COLLISIONS={config.TOTAL_EGO_COLLISIONS} | "
                    f"TOTAL_EGO_TELEPORTS={config.TOTAL_EGO_TELEPORTS} | "
                    f"TOTAL_EGO_EMERGENCY_STOPS={config.TOTAL_EGO_EMERGENCY_STOPS}"
                )

                episode_logger.log(
                    episode=episodes_completed,
                    reward=result.reward,
                    steps=result.steps,
                    end_reason=result.end_reason,
                    total_ego_crashes=config.TOTAL_EGO_CRASHES,
                    total_collision_events=config.TOTAL_COLLISION_EVENTS,
                    total_ego_collisions=config.TOTAL_EGO_COLLISIONS,
                    total_ego_teleports=config.TOTAL_EGO_TELEPORTS,
                    total_ego_emergency_stops=config.TOTAL_EGO_EMERGENCY_STOPS,
                )

                if len(result.rollout) == 0:
                    consecutive_spawn_failures += 1
                    if consecutive_spawn_failures >= MAX_CONSECUTIVE_SPAWN_FAILURES:
                        raise RuntimeError(
                            f"Hit {consecutive_spawn_failures} consecutive spawn failures. "
                            "Check SUMO routes and ego vehicle insertion."
                        )
                    continue

                consecutive_spawn_failures = 0
                batch_results.append(result)
                batch_steps += len(result.rollout)

            update_idx += 1
            stats = ppo_update(actor, critic, optimizer, batch_results)
            update_logger.log(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                update_idx=update_idx,
                episode_start=batch_start_episode + 1,
                episode_end=episodes_completed,
                episodes_in_update=len(batch_rewards),
                rollout_steps=batch_steps,
                mean_reward=f"{np.mean(batch_rewards):.6f}" if batch_rewards else "",
                mean_steps=f"{np.mean(batch_episode_lengths):.6f}" if batch_episode_lengths else "",
                arrived=count_end_reasons(batch_end_reasons, "arrived"),
                ego_crash=count_end_reasons(batch_end_reasons, "ego_crash"),
                abnormal_end=count_end_reasons(batch_end_reasons, "abnormal_end"),
                timeout=count_end_reasons(batch_end_reasons, "timeout"),
                spawn_failed=count_end_reasons(batch_end_reasons, "spawn_failed"),
                policy_loss=f"{stats['policy_loss']:.6f}",
                value_loss=f"{stats['value_loss']:.6f}",
                entropy=f"{stats['entropy']:.6f}",
                loss=f"{stats['loss']:.6f}",
                approx_kl=f"{stats['approx_kl']:.6f}",
                clip_fraction=f"{stats['clip_fraction']:.6f}",
                explained_variance=f"{stats['explained_variance']:.6f}",
                epochs_ran=int(stats["epochs_ran"]),
                samples=int(stats["samples"]),
            )

            print(
                f"Update {update_idx:3d} | "
                f"episodes={batch_start_episode + 1}-{episodes_completed} | "
                f"steps={batch_steps:4d} | "
                f"mean_reward={np.mean(batch_rewards):8.2f} | "
                f"policy_loss={stats['policy_loss']:.4f} | "
                f"value_loss={stats['value_loss']:.4f} | "
                f"approx_kl={stats['approx_kl']:.4f} | "
                f"clip_frac={stats['clip_fraction']:.4f} | "
                f"explained_var={stats['explained_variance']:.4f}"
            )

            if episodes_completed % CHECKPOINT_FREQ == 0:
                checkpoint_path = save_checkpoint(
                    actor,
                    critic,
                    optimizer,
                    episode=episodes_completed,
                    global_step=global_step,
                    update_idx=update_idx,
                )
                print(f"Saved checkpoint to {checkpoint_path}")
    finally:
        try:
            traci.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
