# rl-traffic-agent — Implementation Overview (synced to current code)

This repo trains RL agents to control an **ego vehicle** in a **SUMO** traffic simulation using **TraCI**.

There are **two algorithms** implemented:
- **DQN (value-based)**: `train.py` + `utils.py`
- **PPO (policy-gradient, actor–critic)**: `ppo/train_ppo.py` + `ppo/PPO_actor_critic.py`

If you change the environment, treat this document as the **contract** between:
- the **training loops** (DQN & PPO)
- the **environment**: state / action / reward (`sumo_utils.py`, `Action.py`)
- the **metrics** (incident de-dup + TSV logging: `ego_events.py`, `logger_utils.py`)

Key files:
- Environment plumbing: `sumo_utils.py`, `Action.py`, `config.py`
- DQN: `DQN.py`, `ReplayBuffer.py`, `utils.py`, `train.py`, `validate.py`
- PPO: `ppo/train_ppo.py`, `ppo/PPO_actor_critic.py`, `validate_ppo.py`
- Metrics/logging: `ego_events.py`, `logger_utils.py`

---

## 0) Global configuration snapshot (`config.py`)

Environment-wide constants:
- `SUMO_CONFIG`: default scenario `.sumocfg` file
- `EGO_ID = "ego"`
- Episode horizon: `MAX_STEPS_PER_EPISODE = 1500`
- Route pools:
  - `EGO_ROUTE_POOL`: used for training episodes
  - `VALIDATION_ROUTES`: used by validation sweeps

DQN hyperparameters (used by `train.py`/`utils.py`):
- `GAMMA = 0.99`
- `LR = 1e-3`
- `BATCH_SIZE = 64`
- `BUFFER_CAPACITY = 50_000`
- `MIN_REPLAY_SIZE = 2000`
- `TARGET_UPDATE_FREQ = 500` (**global steps**)
- epsilon schedule over **global steps**:
  - `EPS_START = 1.0` → `EPS_END = 0.05` over `EPS_DECAY = 30000`

PPO hyperparameters live in `ppo/train_ppo.py` (not in `config.py`).

---

## 1) Environment contract (SUMO/TraCI) — `sumo_utils.py`

### 1.1 SUMO lifecycle

The environment is controlled fully by TraCI; time advances only when Python calls:
- `traci.simulationStep()`

**Start** (`start_sumo(...)`):
- picks binary:
  - GUI: `sumo-gui`
  - headless: `config.SUMO_BINARY` (default: `"sumo"`)
- always runs with collision warnings enabled:
  - `--collision.action warn`
  - `--collision.check-junctions true`
  - `--collision.mingap-factor 0`
- writes SUMO log to `sim_logs/sumo.log`
- supports traffic scaling via `--scale <float>`

**Reset** (`reset_sumo(...)`):
- closes TraCI connection (best-effort)
- clears internal per-vehicle caches (notably reward progress cache `_LAST_DISTANCE_BY_VEH_ID`)
- starts SUMO again with the chosen `use_gui`, `sumo_config`, `traffic_scale`

Why this matters:
- per-episode reset is a *full restart*, not a soft reset.
- any state you cache in Python must be cleared across resets.

### 1.2 Spawning the ego vehicle

`spawn_ego(route_id, wait_steps=30) -> (ok: bool, reason: str)`

What it does:
1) Removes a stale ego if it still exists.
2) Calls `traci.vehicle.add(...)` with:
   - `vehID=EGO_ID`
   - route = `route_id`
   - type = `config.EGO_TYPE_ID`
   - lane = `"best"`, pos = `"base"`, speed = `"0"`
3) Steps the simulation up to `wait_steps` times waiting for actual insertion.
4) If GUI is on, sets color and attempts to track the ego in the view.
5) Initializes reward progress baseline:
   - `_LAST_DISTANCE_BY_VEH_ID[EGO_ID] = traci.vehicle.getDistance(EGO_ID)`

Failure modes:
- `invalid_route`: route has no valid connection
- `spawn_blocked`: vehicle never appeared within `wait_steps` (insertion blocked)
- `spawn_error:*`: other TraCI errors

Training code differs by algorithm in how strictly it handles spawn failures:
- DQN training: `utils.run_episode` **does not check** `(ok, reason)`; it checks `ego_exists()` after an extra step.
- PPO training: `run_ppo_episode` checks `(spawn_ok, spawn_reason)` and returns a `spawn_failed:*` end reason.

### 1.3 Observation / state vector (12D)

`get_state(veh_id) -> np.ndarray(shape=(12,), dtype=float32)`

Normalization constants (hard-coded):
- speed scaled by `20.0` (m/s)
- leader gap capped at `100m`
- time headway capped at `10s`
- TLS distance normalized by `200m` (max distance)

Exact feature order (current code):
1) `ego_speed / 20.0`
2) `allowed_speed / 20.0`
3) `min(gap, 100.0) / 100.0`
4) `leader_speed / 20.0`
5) `rel_speed / 20.0` where `rel_speed = ego_speed - leader_speed`
6) `time_headway / 10.0` where `time_headway = gap / max(ego_speed, 0.1)`
7) `tls_dist_norm` (0..1), from `get_tls_info`
8) `tls_red`  (1.0 if next TLS state is 'r')
9) `tls_green` (1.0 if 'g')
10) `tls_yellow` (1.0 if 'y')
11) `lane_pos_norm = lane_pos / lane_length`
12) `route_progress = route_index / max(len(route)-1, 1)`

Edge cases:
- if no leader within 100m: `gap=100`, `leader_speed=0`
- if no next TLS: `tls_dist=1.0` and all TLS flags are `0.0`

### 1.4 Action space (discrete, 5 actions) — `Action.py`

`Action` is an `IntEnum` with fixed indices:
- `0 STRONG_BRAKE`  → `delta_v = -2.0`
- `1 SLOWER`        → `delta_v = -1.0`
- `2 KEEP`          → `delta_v =  0.0`
- `3 FASTER`        → `delta_v = +1.0`
- `4 STRONG_FASTER` → `delta_v = +2.0`

These indices matter:
- DQN outputs **Q-values** with shape `[batch, 5]` and uses `argmax` to pick an index.
- PPO outputs **logits** with shape `[batch, 5]` and samples/argmaxes an index.

### 1.5 How actions are applied (hybrid controller) — `apply_action(...)`

Contract:
- input: `(veh_id, action: Action)`
- output: `delta_v` (the *intended* speed delta for reward smoothness penalty)
- performs: `traci.vehicle.setSpeed(veh_id, new_speed)`

Important: this environment is **NOT** a pure “action = acceleration” simulator.
Your chosen discrete action is filtered by rule-based logic for queues and red lights.

Step-by-step:
1) Get current speed + allowed speed.
2) Compute leader info (gap, leader_speed) and TLS info (dist_norm, red, green, yellow).
3) Convert action → `delta_v` → naive speed target:
   - `new_speed = clamp(current_speed + delta_v, 0, allowed_speed)`
4) Queue override if leader is stopped (`leader_speed < 1.0`):
   - `gap > 5.0` → ensure creeping: `new_speed >= 1.5`
   - `2.5 <= gap <= 4.0` → ensure gentle close-up: `new_speed <= 0.4`
   - `gap < 2.5` → stop: `new_speed = 0.0`
5) Red-light override if `tls_red > 0.5` (TLS distance is normalized by 200m):
   - `tls_dist > 0.04` (~>8m) and `gap > 6m` → keep rolling: `new_speed >= 2.0`
   - `0.02 < tls_dist <= 0.04` (4–8m) and `gap > 5m` → creep: `0.8 <= new_speed <= 2.0`
   - `0.0075 <= tls_dist <= 0.02` (1.5–4m) → near stop: `new_speed <= 0.3`
   - `tls_dist < 0.0075` (<1.5m) → full stop: `new_speed = 0.0`
6) Apply: `traci.vehicle.setSpeed(veh_id, new_speed)`
7) Return `delta_v` (not `new_speed`).

Implication:
- the policy learns under controller overrides.
- reward uses `delta_v` for smoothness, even if the controller clamps/overrides the final speed.

---

## 2) Reward function (detailed) — `sumo_utils.compute_reward(veh_id, delta_v)`

This reward is computed only for **non-terminal** steps (terminal rewards are applied in the episode loops).

High-level design goals:
- discourage wasting time
- reward *actual forward distance traveled* (progress)
- penalize unsafe spacing and risky approaches
- shape behavior near queues and red lights
- keep per-step shaping small to avoid “reward farming”

Implementation details:

### 2.1 Time penalty
Every step:
- `reward -= 0.01`

This pushes the agent to finish sooner and prevents indefinite creeping.

### 2.2 Progress reward (distance delta)
Uses SUMO’s traveled distance:
- `dist_now = traci.vehicle.getDistance(veh_id)`
- `dist_prev = _LAST_DISTANCE_BY_VEH_ID.get(veh_id, dist_now)`
- `delta_dist = max(0.0, dist_now - dist_prev)`
- cache update: `_LAST_DISTANCE_BY_VEH_ID[veh_id] = dist_now`

Progress shaping:
- `reward += min(0.1 * delta_dist, 0.8)`

Notes:
- it rewards **real movement**, not raw speed.
- it clips to `0.8` per step to avoid spikes.
- the cache is initialized during `spawn_ego` and cleared on `reset_sumo`.

### 2.3 Safety distance penalties (gap to leader)
- if `gap < 2.0m`: `reward -= 3.0`
- else if `gap < 2.5m`: `reward -= 1.5`

Unlike earlier versions of the project, these are *small per-step* penalties.

### 2.4 Queue behavior shaping (leader stopped)
Condition: `leader_speed < 1.0`

Penalties:
- stopped too far from queue:
  - if `ego_speed < 0.5 and gap > 8.0`: `reward -= 2.0`
  - elif `ego_speed < 0.5 and gap > 5.0`: `reward -= 1.0`

Bonus:
- good close-but-safe stop:
  - if `2.5 <= gap <= 4.0 and ego_speed < 0.7`: `reward += 0.2`

### 2.5 Approaching leader too fast
- if `rel_speed > 2.0 and gap < 10.0`: `reward -= 1.0`
- if `rel_speed > 4.0 and gap < 6.0`: `reward -= 2.0`

### 2.6 Red light shaping
Condition: `tls_red > 0.5`

Penalties:
- too fast close to red:
  - if `tls_dist < 0.04 and ego_speed > 2.0`: `reward -= 2.0`
  - elif `tls_dist < 0.10 and ego_speed > 5.0`: `reward -= 1.0`
- stopped too far from stop line in red context:
  - if `ego_speed < 0.5 and tls_dist > 0.04 and gap > 6.0`: `reward -= 2.0`
- too close / crossing risk:
  - if `tls_dist < 0.0075 and ego_speed > 0.1`: `reward -= 3.0`

Bonuses:
- creeping closer when red and safe:
  - if `0.02 < tls_dist <= 0.08 and 0.3 <= ego_speed <= 2.5 and gap > 5.0`: `reward += 0.1`
- ideal stop zone near line:
  - if `0.0075 <= tls_dist <= 0.02 and ego_speed < 0.6`: `reward += 0.2`

### 2.7 Yellow caution
- if `tls_yellow > 0.5 and tls_dist < 0.10 and ego_speed > 6.0`: `reward -= 1.0`

### 2.8 Unnecessary stop
Derived helpers:
- `red_far = tls_red > 0.5 and tls_dist > 0.04`
- `leader_far = gap > 8.0`

Penalty:
- if `ego_speed < 0.2 and leader_far and (tls_red < 0.5 or red_far)`: `reward -= 1.0`

Interpretation:
- penalizes stopping when there’s no immediate leader constraint and not in the “close-to-red” context.

### 2.9 Smoothness penalty
- `reward -= 0.03 * abs(delta_v)`

Important:
- this uses the *chosen* `delta_v` (from action) even if `apply_action` overrides speed.

### 2.10 Final clipping
- `reward = clip(reward, -3.0, 1.0)`

That clip strongly bounds gradients/targets for both DQN and PPO.

---

## 3) Episode termination signals

There are two “layers”:

### 3.1 Environment checks (shared helpers)
- `is_arrived()` → ego in `traci.simulation.getArrivedIDList()`
- `is_abnormal_disappearance()` → `(not ego_exists()) and (not is_arrived())`

### 3.2 Terminal rewards (applied by algorithms)
Both algorithms use the same terminal reward values:
- collision/crash: `-30.0`
- arrived: `+20.0`
- abnormal end: `-20.0`
- timeout (horizon reached): DQN uses `-5.0` and pushes a terminal transition; PPO ends with `timeout` (and collects rollout rewards until horizon).

---

## 4) Metrics & incident counting — `ego_events.py`

Why: TraCI can report the same collision/emergency/teleport across multiple steps.

`accumulate_ego_events(...)`:
- keeps per-episode state (`EgoEventState`)
- deduplicates collision events via `collision.collisionID` if available, otherwise `(sim_time, collider, victim)`
- returns per-step deltas (`EgoEventDelta`) for:
  - `TOTAL_COLLISION_EVENTS`
  - `TOTAL_EGO_COLLISIONS`
  - `TOTAL_EGO_TELEPORTS` (max once/episode)
  - `TOTAL_EGO_EMERGENCY_STOPS` (max once/episode)
  - `TOTAL_EGO_CRASHES` (episode-level: 0/1, if ego had any incident)

DQN training and both validations use this de-dup logic.
PPO training currently uses a simpler counter (`update_safety_counters`) which is *not* de-duplicated the same way.

---

## 5) DQN algorithm (training + action selection) — `train.py` + `utils.py`

### 5.1 Model
`DQN.py` defines:
- MLP: `12 → 256 → 256 → 5`
- output: Q-values for each discrete action

### 5.2 Action selection (epsilon-greedy)
In `utils.select_action(policy_net, state, epsilon)`:
- with probability `epsilon`: random `Action`
- else: `argmax(policy_net(state))`

Epsilon schedule is linear in **global steps**:
- `epsilon_by_step(global_step)`

### 5.3 Episode loop ordering (exact)
`utils.run_episode(policy_net, target_net, optimizer, replay_buffer, global_step, route_id)`

Reset + spawn:
1) `reset_sumo()`
2) `spawn_ego(route_id)` (return value ignored)
3) one extra `traci.simulationStep()`
4) if `not ego_exists()` → end = `spawn_failed`

Per-step:
1) choose action via epsilon-greedy (Q argmax)
2) **apply action**: `delta_v = apply_action(EGO_ID, action)`
3) **advance SUMO**: `traci.simulationStep()`; `global_step += 1`
4) incident updates with de-dup: `accumulate_ego_events(...)` → increments `config.TOTAL_*`
5) terminal checks **after step**:
   - crash: if ego in `getCollidingVehiclesIDList()` → terminal `-30`
   - arrived: `+20`
   - abnormal: `-20`
6) non-terminal transition:
   - `next_state = get_state(...)`
   - `reward = compute_reward(..., delta_v)`
   - push replay transition
   - one gradient step via `train_step`
   - hard target-net sync every `TARGET_UPDATE_FREQ` global steps

Timeout:
- after horizon, pushes a terminal transition with reward `-5.0` and `Action.KEEP`.

### 5.4 DQN update (train_step)
`utils.train_step(...)`:
- waits until replay has at least `MIN_REPLAY_SIZE` and `BATCH_SIZE`
- samples batch from `ReplayBuffer`
- computes:
  - `q = policy(states).gather(actions)`
  - `next_q = target(next_states).max()`
  - `target = reward + GAMMA*(1-done)*next_q`
- loss: MSE
- optimizer: Adam from `train.py`

### 5.5 Checkpointing
`train.py` saves weighted checkpoints every 50 episodes:
- `torch.save(policy_net.state_dict(), "dqn_training_5/dqn_ego_episode_{episode+1}.pth")`

---

## 6) PPO algorithm (training + action selection) — `ppo/train_ppo.py`

### 6.1 Model
`ppo/PPO_actor_critic.py`:
- Actor: `12 → 64 → 64 → 5` (outputs logits)
- Critic: `12 → 64 → 64 → 1` (state-value estimate)

### 6.2 Action selection
`select_action(actor, critic, state, deterministic=False)`:
- computes `logits = actor(state)`
- defines `dist = Categorical(logits=logits)`
- action:
  - deterministic=False: `dist.sample()`
  - deterministic=True: `argmax(logits)`
- stores:
  - `log_prob = dist.log_prob(action)`
  - `value = critic(state)`

### 6.3 Data collection: rollout buffer
Per step PPO stores:
- `state, action, log_prob, reward, done, value`

Rollouts can span multiple episodes until:
- at least `ROLLOUT_STEPS` are collected, and
- at least one valid episode exists

### 6.4 Episode loop ordering
`run_ppo_episode(actor, critic, global_step)`:

1) `reset_sumo(use_gui=False)`
2) pick random route from `EGO_ROUTE_POOL`
3) `spawn_ego(route_id)` and if fail → end = `spawn_failed:*`
4) per-step (up to `MAX_STEPS_PER_EPISODE`):
   - choose action (sample)
   - `delta_v = apply_action(...)`
   - `traci.simulationStep()` and increment `global_step`
   - update safety counters via `update_safety_counters()` (note: simpler than `ego_events`)
   - terminal checks (crash/arrived/abnormal)
   - otherwise get `next_state` and compute shaped reward `compute_reward(...)`
   - store transition in rollout

Terminal rewards match DQN’s ones:
- crash: -30
- arrived: +20
- abnormal: -20

At episode end:
- computes `last_value` from critic if ego still exists

### 6.5 PPO update math
`ppo_update(...)`:
- computes per-episode GAE (`compute_gae`) producing:
  - returns
  - advantages (normalized)
- PPO clipped objective:
  - `ratio = exp(new_logp - old_logp)`
  - `policy_loss = -mean(min(ratio*A, clip(ratio)*A))`
- value loss: MSE
- entropy bonus
- gradient clipping: `clip_grad_norm_(..., MAX_GRAD_NORM)`
- early stop by KL if average approx_kl > `TARGET_KL`

### 6.6 PPO checkpointing
`saved every CHECKPOINT_FREQ episodes` into `ppo/ppo_training/` as a dict with:
- `actor_state_dict`, `critic_state_dict`, `optimizer_state_dict`
- `episode`, `global_step`, `update_idx`

---

## 7) Validation (DQN and PPO)

### 7.1 Shared validation TSV format
`logger_utils.ValidationEpisodeLogger` writes TSV with columns:
- `timestamp`
- `combo_episode` (route×scale index)
- `route_id`
- `traffic_scale`
- `reward`
- `steps`
- `end_reason`
- `ego_crash`
- `ego_collision_events`
- `ego_teleport`
- `ego_emergency_stop`
- `sim_time_end`
- `model_path`
- `sumo_config`

### 7.2 DQN validation
- CLI: `validate.py`
- runner: `utils.validate_model_on_routes(...)`
- per-episode: `utils.run_loaded_model_on_route(...)` which:
  - loads `DQN(12,5)` weights
  - runs greedy `argmax(Q)` actions
  - counts incidents with `accumulate_ego_events` (de-duplicated)

### 7.3 PPO validation
- CLI: `validate_ppo.py`
- runner: `utils.validate_ppo_model_on_routes(...)`
- per-episode: `utils.run_loaded_ppo_model_on_route(...)` which:
  - loads PPO checkpoint dict and uses `actor_state_dict`
  - runs deterministic argmax by default (optional stochastic sampling)
  - counts incidents with `accumulate_ego_events` (same as DQN validation)

---

## 8) Debugging helpers

`gui_manual_step.py`:
- starts SUMO-GUI
- spawns ego
- advances one tick per Enter press

Rationale:
- when SUMO is TraCI-controlled, using the GUI’s step button can conflict with TraCI.

---

## 9) “Keep in sync” checklist (doc drift prevention)

If you change code, re-check:
- **State dim** is 12 everywhere:
  - `sumo_utils.get_state`
  - DQN: `DQN(12, 5)`
  - PPO: `Actor(12, 5)` and `Critic(12)`
- **Action indices** match `Action(IntEnum)` ordering
- **Action application** overrides (queues + red lights) still align with reward shaping
- Reward’s **progress cache** is reset properly on reset/spawn (`_LAST_DISTANCE_BY_VEH_ID`)
- Terminal checks happen **after** `traci.simulationStep()`
- DQN target-net sync uses **global_step**, not episode
- Validation TSV formats remain identical for easy comparison
