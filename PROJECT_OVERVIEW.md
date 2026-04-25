# rl-traffic-agent — Implementation Overview

This repo trains a **DQN (Deep Q-Network)** agent to control an **ego vehicle** in a **SUMO** traffic simulation using **TraCI**.

The project is intentionally lightweight: a few Python modules implement
- SUMO start/reset + observation/reward/action application
- DQN model + replay buffer
- training loop
- logging
- (crucially) **correct ego-event accounting** (collisions/teleports/emergency stops) without double-counting.

---

## 1) Core architecture (data/control flow)

At a high level:

1. `train.py` starts SUMO (TraCI), creates networks/optimizer/buffer.
2. For each episode, it calls `utils.run_episode(...)`.
3. `utils.run_episode(...)`:
   - resets SUMO
   - spawns the ego vehicle
   - loops step-by-step:
     - chooses an action (epsilon-greedy)
     - applies the action via TraCI
     - advances SUMO by **exactly one** `traci.simulationStep()`
     - reads next state + reward + terminal conditions
     - pushes transition to replay buffer
     - performs a DQN training step (if buffer is warm)
     - updates the target network periodically
     - updates event counters via `ego_events.py`
4. `logger_utils.EpisodeLogger` appends per-episode summaries to `logs/*.tsv`.

**Key invariants**:
- When SUMO is started via TraCI, **Python is the controller**. Time must advance via `traci.simulationStep()` from Python.
- Episode resets must be **clean** (restart SUMO) and must preserve GUI/headless mode.
- Ego incident counters must be **de-duplicated** across steps (SUMO may report same incident many steps).

---

## 2) Module-by-module breakdown

### `config.py`
**Purpose**: central configuration for training and SUMO.

Contains:
- SUMO config path (`SUMO_CONFIG`) and binary (`SUMO_BINARY`)
- DQN hyperparameters (`GAMMA`, `LR`, `BATCH_SIZE`, etc.)
- epsilon schedule (`EPS_START`, `EPS_END`, `EPS_DECAY`)
- ego vehicle identity (`EGO_ID`, `EGO_TYPE_ID`) and possible routes (`EGO_ROUTE_POOL`)
- global cumulative counters:
  - `TOTAL_EGO_CRASHES`
  - `TOTAL_COLLISION_EVENTS`
  - `TOTAL_EGO_COLLISIONS`
  - `TOTAL_EGO_TELEPORTS`
  - `TOTAL_EGO_EMERGENCY_STOPS`

**Most important moments**:
- These counters are **cumulative** (not reset per episode). If you want per-episode counters, log deltas or reset explicitly.
- Resuming training properly often requires resuming schedule state too (`global_step`, epsilon), not only model weights.

---

### `Action.py`
**Purpose**: defines the discrete action space.

- `Action` is an `IntEnum` with 5 discrete actions:
  - `STRONG_BRAKE`, `SLOWER`, `KEEP`, `FASTER`, `STRONG_FASTER`
- `ACTION_TO_DELTA_V` maps each action to a speed delta (m/s).

**Most important moments**:
- This action space is **speed-setpoint based** (not acceleration). The agent changes speed by `delta_v` and clamps at 0.
- If you expand this later (lane changes etc.), be careful: invalid actions can increase teleports/emergency stops.

---

### `DQN.py`
**Purpose**: a small MLP used as Q-network.

- Architecture: `state_dim -> 128 -> 128 -> action_dim` with ReLU.

**Most important moments**:
- You use **two networks** in training:
  - `policy_net` (trained)
  - `target_net` (periodic copy for stable TD targets)

---

### `ReplayBuffer.py`
**Purpose**: experience replay.

- Stores tuples `(state, action, reward, next_state, done)` in a `deque(maxlen=capacity)`.
- `sample(batch_size)` returns NumPy arrays of each field.

**Most important moments**:
- Buffer is not persisted by default. If you resume training from a checkpoint, training is **not identical** unless you also restore the buffer.

---

### `sumo_utils.py`
**Purpose**: all SUMO/TraCI environment interaction (start/reset/spawn/state/reward).

Key functions:

#### `start_sumo(use_gui=False, log_dir="sim_logs", quiet_console=False, hide_warnings=False)`
- Builds a SUMO command and calls `traci.start(cmd)`.
- Tracks GUI mode in global `CURRENT_USE_GUI`.

**Critical detail**: `use_gui=True` uses `sumo-gui`, otherwise `SUMO_BINARY` (typically `sumo`).

#### `_traci_is_connected()`
- Best-effort check: tries `traci.simulation.getTime()`.

#### `reset_sumo(use_gui=None)`
- Restarts SUMO cleanly.
- If `use_gui is None`, restarts using `CURRENT_USE_GUI` (same mode as current run).

**Most important moments**:
- This avoids a major class of issues where GUI mode accidentally flips after reset and TraCI disconnects.

#### `spawn_ego(route_id, wait_steps=30)`
- Adds ego once via `traci.vehicle.add(...)`.
- Steps simulation until ego appears in `traci.vehicle.getIDList()`.
- Returns `(ok, reason)` where reason can be `invalid_route`, `spawn_blocked`, etc.

**Most important moments**:
- SUMO may refuse insertion (blocked). This function waits and fails cleanly.
- If GUI is used, it tracks the ego vehicle and colors it red.

#### Observation helpers
- `get_leader_info(...)`, `get_tls_info(...)`, `get_state(...)`.

The state vector is an 8D normalized float array:
1. ego speed
2. gap to leader
3. leader speed
4. relative speed
5. TLS distance
6. TLS is red
7. TLS is green
8. TLS is yellow

#### Action application
- `apply_action(veh_id, action)` reads current speed, adds delta, clamps at 0, sets speed.

#### Reward
- `compute_reward(...)` is a shaped reward:
  - + speed progress
  - penalties for low gap / closing in fast
  - penalty for approaching red light fast
  - smoothness penalty on |delta_v|

#### Terminal checks
- `is_arrived()` uses `traci.simulation.getArrivedIDList()`
- `is_abnormal_disappearance()` is `not ego_exists()` and not arrived

---

### `ego_events.py` (CRITICAL: correct counting)
**Purpose**: correct ego-related event accounting without double counting.

Why needed:
- TraCI APIs like `getCollisions()`, `getStartingTeleportIDList()`, etc. can report the same incident for multiple steps.
- Counting `+= len(events)` per step will inflate totals.

#### `EgoEventState`
Per-episode state:
- `seen_collision_keys`: set of collisions already counted this episode
- `ego_incident_counted`: whether we already counted a crash for this episode
- `ego_teleport_counted`, `ego_emergency_counted`: prevent repeated counting

#### `accumulate_ego_events(...) -> (state, delta)`
Inputs are the per-step event lists plus `sim_time`.
Outputs:
- updated `state`
- `delta` with increments for:
  - total collisions (system-wide)
  - ego collisions (events involving ego)
  - ego crashes (at most 1 per episode)
  - ego teleports (at most 1 per episode)
  - ego emergency stops (at most 1 per episode)

**Most important moments / semantics**:
- `TOTAL_EGO_COLLISIONS` counts **events** (can be >1 per episode).
- `TOTAL_EGO_CRASHES` counts **episodes with any ego incident** (0/1 per episode), where incident ∈ {collision, teleport, emergency stop}.

**Potential follow-up improvement**:
- Collision de-dup uses `collisionID` if available, else `(sim_time, collider, victim)`.
  - If your SUMO doesn’t provide `collisionID` and emits the *same* collision across multiple times, this fallback may still count multiple times.
  - If this happens in your runs, change the fallback key to ignore time (e.g., `(collider, victim)` optionally ordered).

---

### `utils.py`
**Purpose**: all RL glue: epsilon schedule, action selection, TD update training step, and episode rollout.

#### `epsilon_by_step(global_step)`
- Linear anneal from `EPS_START` to `EPS_END` over `EPS_DECAY` steps.

#### `select_action(policy_net, state, epsilon)`
- epsilon-greedy:
  - explore: `random.choice(list(Action))`
  - exploit: argmax over Q-values

#### `train_step(policy_net, target_net, optimizer, replay_buffer)`
- Returns `None` until replay buffer is warm (`MIN_REPLAY_SIZE`, `BATCH_SIZE`).
- Standard DQN TD target:
  - `q = policy(states)[action]`
  - `target = r + gamma * (1-done) * max_a target(next_state)`
  - MSE loss

**Most important moments**:
- Uses `target_net` only for the bootstrap value, under `torch.no_grad()`.

#### `run_episode(policy_net, target_net, optimizer, replay_buffer, global_step)`
Episode procedure:
1. Choose `route_id` randomly from `EGO_ROUTE_POOL`.
2. `reset_sumo()` (clean restart)
3. `spawn_ego(route_id)`
4. `traci.simulationStep()` once more to ensure insertion
5. If ego missing: return `spawn_failed`
6. Initialize `event_state = EgoEventState()` (IMPORTANT: reset per episode)
7. For each environment step:
   - compute epsilon, select action, apply action
   - `traci.simulationStep()`
   - collect events lists from TraCI
   - call `accumulate_ego_events(...)` and update global counters
   - handle terminal conditions:
     - ego crash (collision)
     - arrived
     - abnormal disappearance
   - otherwise: compute next_state & shaped reward, push to buffer, train
   - periodic target net update every `TARGET_UPDATE_FREQ` global steps

**Most important moments**:
- Correctness depends on the exact ordering:
  - action is applied *before* stepping
  - events are read *after* stepping
- Ego events are counted via `ego_events.py` deltas:
  - prevents per-step double counting

#### `run_loaded_model_on_route(model_path, route_id, use_gui=False, max_steps=None)`
- Loads a trained model and runs a deterministic evaluation episode.

**Critical warning**:
- When TraCI controls SUMO, do **not** click SUMO-GUI’s Step button.

---

### `logger_utils.py`
**Purpose**: append-only episode logging.

- `EpisodeLogger` writes a TSV header if file doesn’t exist.
- `log(...)` appends one line with:
  - timestamp, episode, reward, steps, end_reason
  - cumulative totals from `config` for incidents

**Most important moments**:
- It logs cumulative totals, not per-episode deltas.
- This is fine if you interpret them as “total so far”. If you need per-episode events, log the delta per episode too.

---

### `gui_manual_step.py`
**Purpose**: safe manual stepping in GUI.

- Starts sumo-gui via `start_sumo(use_gui=True)`.
- Spawns ego.
- Steps only when you press Enter in the terminal.

**Most important moments**:
- In TraCI-controlled mode, the GUI Step button competes with Python and often causes disconnects.

---

## 3) Training entrypoint: `train.py`

`train.py` wiring:
- starts TraCI/SUMO once (`start_sumo(use_gui=False)`)
- creates `policy_net` & `target_net`
- optional resume:
  - load policy weights
  - load optimizer state (if provided)
- for each episode:
  - call `run_episode(...)`
  - log cumulative counters to TSV
  - optional checkpoint save

**Most important moments / pitfalls**:
1. **Resume training is currently partial** unless you also restore:
   - `global_step` (epsilon schedule)
   - replay buffer contents (optional but affects learning)
2. If you expect “continue exactly”, you need to checkpoint: weights + optimizer + metadata.

---

## 4) Hard-earned correctness rules (the stuff that breaks most often)

1. **Do not use SUMO-GUI Step button while TraCI is connected**.
   - Always advance with `traci.simulationStep()` from Python.

2. **Reset SUMO carefully**.
   - Restart in the same GUI/headless mode.

3. **Never count `len(getCollisions())` per step without de-dup**.
   - Use stable keys (`collisionID`) or a robust fallback.

4. **Decide and document your counter semantics**.
   - This repo uses:
     - `TOTAL_EGO_COLLISIONS`: collision events involving ego
     - `TOTAL_EGO_CRASHES`: episodes with any ego incident (collision/teleport/emergency)

---

## 5) Suggested next improvements (optional, but high value)

- Add per-episode event deltas to the logger (in addition to cumulative totals).
- Persist and restore training metadata:
  - global_step
  - replay buffer
  - RNG state
- Make collision de-dup fallback not depend on `sim_time` if your SUMO doesn’t expose `collisionID`.

---

## 6) Very detailed training process (what happens line-by-line conceptually)

This section describes, in detail, what happens during training when you run `python train.py`.

### 6.1 Initialization (`train.py`)

1. **Imports + configuration**
   - `config.py` defines hyperparameters (gamma, batch size, epsilon schedule), SUMO configuration path, and global counters.

2. **(Optional) Resume configuration**
   - `RESUME_PATH` (model weights) and `RESUME_OPTIM_PATH` (optimizer state) control whether training starts from scratch or continues.
   - `START_EPISODE` controls the episode index for logging/checkpoint naming.

   Important caveat:
   - In the current code, `global_step` is always set to `0` on start. That means the **epsilon schedule restarts** unless you explicitly restore `global_step`.

3. **Start SUMO / TraCI exactly once for the process**
   - `start_sumo(use_gui=False)` starts SUMO in TraCI-controlled mode.
   - Even though SUMO is started once here, each episode calls `reset_sumo()` (which closes + restarts SUMO) for a clean episodic reset.

   Most important moment:
   - Once TraCI is connected, SUMO time must advance from Python via `traci.simulationStep()`.

4. **Create neural networks**
   - `policy_net = DQN(state_dim=8, action_dim=len(Action))`
   - `target_net = DQN(...)`

5. **Load weights if resuming**
   - If `RESUME_PATH` is set:
     - `policy_net.load_state_dict(torch.load(...))`

6. **Synchronize target network**
   - `target_net.load_state_dict(policy_net.state_dict())`
   - `target_net.eval()` (target is not trained by gradients)
   - `policy_net.train()`

   Most important moment:
   - DQN stability depends heavily on the target network being a *lagged* copy.

7. **Create optimizer**
   - `optimizer = Adam(policy_net.parameters(), lr=LR)`

8. **Load optimizer state if resuming**
   - If `RESUME_OPTIM_PATH` is set and exists:
     - `optimizer.load_state_dict(torch.load(...))`

   Most important moment:
   - If you resume weights but not optimizer state, Adam’s internal moments reset and you effectively “switch optimizers”. It can still work, but it’s not a true continuation.

9. **Create replay buffer**
   - `replay_buffer = ReplayBuffer(BUFFER_CAPACITY)`

   Important caveat:
   - Replay buffer state is not restored in the current code. So “resume training” is currently: **resume weights (+ maybe optimizer)** but **not** buffer history.

10. **Create logger**
   - `EpisodeLogger(default_episode_log_path())` writes a TSV file under `logs/`.

11. **Main episode loop**
   - For each episode:
     - call `utils.run_episode(...)`
     - log episode summary
     - checkpoint occasionally

---

### 6.2 Episode lifecycle (`utils.run_episode`)

Each call to `run_episode(policy_net, target_net, optimizer, replay_buffer, global_step)` runs exactly one episode.

**Inputs**
- the current networks and optimizer
- replay buffer (shared across episodes)
- `global_step` (used for epsilon schedule + target update timing)

**Outputs**
- `episode_reward`
- `episode_steps`
- updated `global_step`
- `end_reason` (e.g., `arrived`, `ego_crash`, `timeout`, ...)

#### Step A — Choose a route
- `route_id = random.choice(EGO_ROUTE_POOL)`

This gives route diversity (multiple initial conditions).

#### Step B — Hard reset SUMO
- `reset_sumo()`

In this repo, reset is implemented as:
- close TraCI connection
- restart SUMO

This is slower than `traci.load(...)`, but it is simple and avoids a lot of subtle state bugs.

**Most important moment**:
- reset must preserve GUI/headless mode. `reset_sumo(use_gui=None)` uses the last started mode.

#### Step C — Spawn the ego
- `spawn_ego(route_id)` adds a vehicle with ID `EGO_ID` and then steps the simulation until insertion.

Important detail:
- insertion can fail (blocked). `spawn_ego` returns `(ok, reason)` but `run_episode` currently calls it without checking the tuple. The subsequent `ego_exists()` check is the real gate.

#### Step D — One extra simulation step
- `traci.simulationStep()`

This ensures the ego is fully inserted and TraCI state queries (speed, leader, TLS) won’t crash.

#### Step E — Initialize episode state and event de-dup state
- `event_state = EgoEventState()`

This is **per-episode** state used to prevent counting the same collision/teleport across many ticks.

---

### 6.3 Per-step loop (the heart of training)

This loop runs up to `MAX_STEPS_PER_EPISODE` iterations.

#### 1) Compute epsilon
- `epsilon = epsilon_by_step(global_step)`

The schedule (linear anneal) is:
- start at `EPS_START`
- move toward `EPS_END` over `EPS_DECAY` global steps

#### 2) Select an action (epsilon-greedy)
- With probability `epsilon`: choose a random `Action`
- Otherwise: choose `argmax_a Q_policy(state, a)`

Important detail:
- action indices map to `Action` enum values in `Action.py`.

#### 3) Apply the action to SUMO (control)
- `delta_v = apply_action(EGO_ID, action)`

`apply_action` does:
- read current ego speed from TraCI
- add action delta (`ACTION_TO_DELTA_V[action]`)
- clamp speed to `>= 0`
- call `traci.vehicle.setSpeed(...)`

#### 4) Advance simulation by exactly one tick
- `traci.simulationStep()`
- `global_step += 1`

**Most important moment**:
- This repo assumes you advance simulation only here. If you click GUI Step while running under TraCI, SUMO and Python will fight and you can get disconnects / time=-1.

#### 5) Read events and update counters (de-duplicated)
This is the crucial safety logic for your metrics.

- Read lists from TraCI:
  - `colliding_ids = traci.simulation.getCollidingVehiclesIDList()`
  - `collision_events = traci.simulation.getCollisions()`
  - `teleport_ids = traci.simulation.getStartingTeleportIDList()`
  - `emergency_ids = traci.simulation.getEmergencyStoppingVehiclesIDList()`

- Pass them into `accumulate_ego_events(...)`:
  - maintains `event_state` (seen collisions, whether teleport/emergency already counted)
  - returns `delta` increments for counters

- Add deltas into `config.TOTAL_*` counters.

**Most important moment**:
- We intentionally do **not** do `+= len(collision_events)` per step. That would inflate counts.

#### 6) Terminal checks (episode ends here)
After updating counters, the episode checks terminals in this order:

1. **Collision involving ego** (hard failure)
   - `ego_crashed = EGO_ID in colliding_ids`

   If true:
   - assign terminal reward (currently `-30`)
   - push terminal transition into replay buffer with `done=1.0`
   - do one training step (if buffer warm)
   - return `(episode_reward, steps, global_step, 'ego_crash')`

2. **Arrived** (success)
   - `is_arrived()` uses `traci.simulation.getArrivedIDList()`

   If true:
   - reward `+20`
   - push terminal transition
   - train step
   - return `('arrived')`

3. **Abnormal disappearance** (teleport/remove/etc.)
   - `is_abnormal_disappearance()` is `(not ego_exists()) and (not is_arrived())`

   If true:
   - reward `-20`
   - push terminal transition
   - train step
   - return `('abnormal_end')`

**Most important moment**:
- Terminal transitions are stored in the replay buffer, which is required for correct Q-learning with terminal states.

#### 7) Non-terminal transition
If no terminal condition triggered:

- `next_state = get_state(EGO_ID)`
- `reward = compute_reward(EGO_ID, delta_v)`
- `done = 0.0`

Then:
- push to replay buffer
- run `train_step(...)`
- update target network periodically
- accumulate episode reward
- `state = next_state`

---

### 6.4 DQN training step details (`utils.train_step`)

This function implements classic DQN.

#### Replay buffer warmup
Training returns `None` until:
- `len(replay_buffer) >= MIN_REPLAY_SIZE`
- and `>= BATCH_SIZE`

This avoids training on a tiny, highly correlated dataset.

#### Batch tensorization
The sampled batch is converted to torch tensors on `DEVICE`.

#### Q-value and target computation
- Current Q(s,a):
  - `q_values = policy_net(states).gather(1, actions)`

- Target values (bootstrap):
  - `next_q = target_net(next_states).max(dim=1)`
  - `target = reward + gamma * (1-done) * next_q`

Important detail:
- `target_net` call is inside `torch.no_grad()` so gradients do not flow into target.

#### Loss and optimizer step
- MSE loss between current and target
- Backprop on policy network only

---

### 6.5 Target network update

In `run_episode`:
- every `TARGET_UPDATE_FREQ` global steps:
  - `target_net.load_state_dict(policy_net.state_dict())`

**Most important moment**:
- This is essential for training stability.
- If you remove it, learning often diverges.

---

### 6.6 Logging and checkpointing (`train.py`)

After each episode:
- `EpisodeLogger.log(...)` writes:
  - reward, steps, end_reason
  - cumulative totals: collisions/crashes/teleports...

Checkpointing:
- every 50 episodes:
  - saves only `policy_net.state_dict()` into `dqn_training_2/`.

Important caveats:
- Optimizer state is not checkpointed here.
- `global_step` is not checkpointed.
- Replay buffer is not checkpointed.

If you need *true* resume-training, you typically checkpoint a dict containing:
- model weights
- optimizer state
- episode index
- global_step
- (optionally) replay buffer

---

### 6.7 Termination modes you’ll see in logs

`end_reason` can be:
- `ego_crash`: ego was in `getCollidingVehiclesIDList()` after a sim step
- `arrived`: ego in arrived list
- `abnormal_end`: ego disappeared without arriving (teleport/remove)
- `timeout`: exceeded `MAX_STEPS_PER_EPISODE`
- `spawn_failed`: ego never inserted

---

### 6.8 The most important training correctness rules (re-stated)

1. **One controller steps SUMO**
   - While running `train.py` / TraCI, do not click SUMO-GUI step.

2. **Events must be de-duplicated**
   - SUMO may report the same collision/teleport across multiple steps.
   - Use `ego_events.py` state + delta pattern.

3. **Define exact counter semantics**
   - In this repo:
     - `TOTAL_EGO_COLLISIONS` = number of ego-involved collision events (dedup)
     - `TOTAL_EGO_CRASHES` = number of episodes where ego had any incident (0/1 per episode)

4. **Resume-training requires restoring schedules**
   - Restore `global_step` if you care about epsilon/target timing continuity.


