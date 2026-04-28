# rl-traffic-agent — Implementation Overview (synced to current code)

This repo trains a **DQN (Deep Q-Network)** agent to control an **ego vehicle** in a **SUMO** traffic simulation using **TraCI**.

If you’re changing the environment, treat this document as the **contract** between:
- the **training loop** (`train.py`, `utils.py`)
- the **environment** (state/action/reward in `sumo_utils.py`)
- the **metrics** (incident de-dup in `ego_events.py`, episode TSV via `logger_utils.py`)

Key files:
- **Training**: `train.py`, `utils.py`
- **Environment**: `sumo_utils.py`, `Action.py`
- **Model**: `DQN.py`, `ReplayBuffer.py`
- **Metrics/logging**: `ego_events.py`, `logger_utils.py`, `config.py`

---

## 0) Quick config snapshot (from `config.py`)

These values heavily shape learning dynamics:
- Episodes / horizon: `NUM_EPISODES=500`, `MAX_STEPS_PER_EPISODE=1500`
- DQN: `GAMMA=0.99`, `LR=1e-3`, `BATCH_SIZE=64`
- Replay: `BUFFER_CAPACITY=50000`, `MIN_REPLAY_SIZE=2000`
- Target net sync: `TARGET_UPDATE_FREQ=500` (**global steps**)
- Exploration schedule (linear in **global steps**):
  - `EPS_START=1.0 → EPS_END=0.05` over `EPS_DECAY=30000`

---

## 1) End-to-end training flow (`python train.py`)

### 1.1 Startup (`train.py`)

`train.py` does the plumbing and then repeatedly calls `utils.run_episode(...)`.

What gets created:
1) Start SUMO once:
   - `start_sumo(use_gui=False)`
   - Each episode will still restart SUMO via `reset_sumo()` (see below).

2) Define dimensions (must match `sumo_utils.get_state` and `Action`):
   - `state_dim = 12`
   - `action_dim = len(Action)` (5)

3) Create the networks:
   - `policy_net = DQN(state_dim, action_dim).to(DEVICE)`
   - `target_net = DQN(state_dim, action_dim).to(DEVICE)`
   - `target_net.load_state_dict(policy_net.state_dict())`
   - `target_net.eval()` and `policy_net.train()`

4) Optimizer:
   - `optim.Adam(policy_net.parameters(), lr=LR)`
   - Optional resume: `RESUME_OPTIM_PATH` can restore optimizer state.

5) Replay buffer:
   - `ReplayBuffer(BUFFER_CAPACITY)`

6) Global training step:
   - `global_step = 0` at the start of the script.
   - Important: epsilon and target-net updates depend on `global_step`, so if you restart the script without restoring it, exploration restarts.

7) Episode logger:
   - `EpisodeLogger(default_episode_log_path())` writes `logs/train_*.tsv`
   - The TSV includes `route` and cumulative incident counters from `config.TOTAL_*`.

### 1.2 Episode loop

For each episode:
- choose `route_id = random.choice(EGO_ROUTE_POOL)`
- call `run_episode(policy_net, target_net, optimizer, replay_buffer, global_step, route_id)`
- write one TSV row with: episode reward, steps, end reason, route, and cumulative incident counters
- every 50 episodes: save `policy_net` weights (currently into `dqn_training_4/`)

Resume caveat:
- weights-only resume **is not** a full continuation
- to fully continue the same training run you’d want to persist at least:
  - policy weights
  - optimizer state
  - `global_step`
  - (optionally) replay buffer contents

---

## 2) Episode rollout (`utils.run_episode`) — exact ordering (important)

Signature:
- `run_episode(policy_net, target_net, optimizer, replay_buffer, global_step, route_id=False)`

### 2.1 Reset + spawn

At the start of every episode:
1) `reset_sumo()`
   - closes the current TraCI connection (if alive)
   - starts SUMO again with the same GUI/headless mode as the current run

2) `spawn_ego(route_id)`
   - adds `EGO_ID` on the given `route_id`
   - internally steps SUMO up to `wait_steps` to wait for real insertion
   - returns `(success, reason)` in `{spawned, invalid_route, spawn_blocked, spawn_error}`
   - Note: `run_episode` currently *does not* check this return value; it checks `ego_exists()` afterwards.

3) One extra `traci.simulationStep()`
   - `run_episode` advances one more step “to be safe” after spawn.

If `ego_exists()` is still false ⇒ episode ends with:
- `(reward=0.0, steps=0, end_reason="spawn_failed")`

### 2.2 Per-step loop (one TraCI tick per RL step)

For `step in range(MAX_STEPS_PER_EPISODE)`:

1) **Exploration rate** (linear schedule):
   - `epsilon = epsilon_by_step(global_step)`

2) **Action selection**:
   - epsilon-greedy over `policy_net(state)`
   - random action is `random.choice(list(Action))`

3) **Apply action BEFORE stepping SUMO**:
   - `delta_v = apply_action(EGO_ID, action)`
   - (details in section 5)

4) **Advance SUMO by one tick**:
   - `traci.simulationStep()`
   - `global_step += 1`

5) **Update incident counters (de-duplicated)**:
   - read TraCI lists:
     - `getCollidingVehiclesIDList()`, `getCollisions()`
     - `getStartingTeleportIDList()`
     - `getEmergencyStoppingVehiclesIDList()`
   - call `accumulate_ego_events(...)`
   - increment `config.TOTAL_*` by the returned per-step deltas

6) **Terminal checks (after stepping):**
   - ego collision (`EGO_ID in getCollidingVehiclesIDList`) ⇒ terminal `reward=-30`, `end_reason="ego_crash"`
   - arrived (`EGO_ID in getArrivedIDList`) ⇒ terminal `reward=+20`, `end_reason="arrived"`
   - abnormal disappearance (`not ego_exists()` and not arrived) ⇒ terminal `reward=-20`, `end_reason="abnormal_end"`

7) **If not terminal: learn from a normal transition**
   - `next_state = get_state(EGO_ID)`
   - `reward = compute_reward(EGO_ID, delta_v)`
   - push to replay buffer
   - do exactly one `train_step(...)` (may be skipped until buffer is warm)
   - update target net when `global_step % TARGET_UPDATE_FREQ == 0`

Timeout:
- after `MAX_STEPS_PER_EPISODE` steps, if ego still exists:
  - push a terminal transition with `reward=-5`, `done=1.0`, `action=Action.KEEP`
  - return `end_reason="timeout"`

---

## 3) DQN update (`utils.train_step`) — what’s optimized

Training starts only after replay is “warm”:
- `len(replay_buffer) >= MIN_REPLAY_SIZE` and `>= BATCH_SIZE`

Batch format (`ReplayBuffer.sample`):
- `(states, actions, rewards, next_states, dones)`

Targets (standard DQN with a target network):
- `q = policy_net(states).gather(1, actions)`
- `next_q = target_net(next_states).max(dim=1)`
- `target = reward + GAMMA * (1-done) * next_q`

Loss:
- MSE loss (`nn.MSELoss`) between `q` and `target`

Target net sync:
- hard copy in `run_episode` when `global_step % TARGET_UPDATE_FREQ == 0`

Model architecture (`DQN.py`):
- MLP: `12 → 256 → 256 → 5` with ReLU activations

---

## 4) State vector (12D) — the main environment contract (`sumo_utils.get_state`)

`get_state(veh_id)` returns a `np.float32` vector of length 12.

**Normalization constants** (important when changing scenarios):
- speeds are scaled by `20.0` (m/s)
- gap is capped at `100m` (leader search is also limited to `100m`)
- time headway is capped at `10s`
- traffic light distance is normalized by `200m` and capped at that

Exact order:

1) `ego_speed / 20.0`
2) `allowed_speed / 20.0`
3) `min(gap, 100.0) / 100.0`
4) `leader_speed / 20.0`
5) `rel_speed / 20.0`, where `rel_speed = ego_speed - leader_speed`
6) `time_headway / 10.0`, where `time_headway = gap / max(ego_speed, 0.1)` (capped at 10s)
7) `tls_dist_norm` (nearest next TLS distance / 200m, capped)
8) `tls_red` (1.0 if red else 0.0)
9) `tls_green` (1.0 if green else 0.0)
10) `tls_yellow` (1.0 if yellow else 0.0)
11) `lane_pos_norm = lane_pos / lane_length`
12) `route_progress = route_index / max(len(route)-1, 1)`

Edge-case behavior worth knowing:
- If no leader within `100m`: `gap=100`, `leader_speed=0` (via `get_leader_info`)
- If no TLS ahead: `tls_dist_norm=1.0` and all TLS flags are `0.0`

---

## 5) Actions — discrete policy output with rule-based safety filtering

### 5.1 Discrete action space (`Action.py`)

`Action` is an `IntEnum` with 5 actions:
- `STRONG_BRAKE (0)`
- `SLOWER (1)`
- `KEEP (2)`
- `FASTER (3)`
- `STRONG_FASTER (4)`

Mapping to “suggested speed delta” (m/s per step):
- `-2.0, -1.0, 0.0, +1.0, +2.0`

### 5.2 Applying the action (`sumo_utils.apply_action`)

`apply_action(veh_id, action)` implements a **hybrid controller**:
1) Convert action to delta-v and compute a naïve target speed:
   - `new_speed = clamp(current_speed + delta_v, 0, allowed_speed)`

2) If the leader is stopped (`leader_speed < 1.0`), override to manage queues:
   - `gap > 5m` ⇒ force creep forward (`new_speed >= 1.5`)
   - `2.5m ≤ gap ≤ 4m` ⇒ force very slow close-up (`new_speed ≤ 0.4`)
   - `gap < 2.5m` ⇒ full stop (`new_speed = 0.0`)

3) If the next traffic light is red (`tls_red > 0.5`), override to approach but not cross:
   - `tls_dist > 0.04` (~>8m) and `gap > 6m` ⇒ keep rolling (`new_speed >= 2.0`)
   - `0.02 < tls_dist ≤ 0.04` (4–8m) and `gap > 5m` ⇒ slow creep (`0.8 ≤ new_speed ≤ 2.0`)
   - `0.0075 ≤ tls_dist ≤ 0.02` (1.5–4m) ⇒ almost stop (`new_speed ≤ 0.3`)
   - `tls_dist < 0.0075` (<1.5m) ⇒ stop (`new_speed = 0.0`)

4) Apply to SUMO:
   - `traci.vehicle.setSpeed(veh_id, new_speed)`

Return value:
- `apply_action` returns the chosen `delta_v` (the *intended* speed delta), not the resulting `new_speed`.

Important implication:
- This is **not** a “pure RL” environment. Actions are filtered by rule-based logic.
- When debugging learning, always remember the network learns under these overrides.

---

## 6) Reward function — shaped step reward + terminal rewards

### 6.1 Step reward (`sumo_utils.compute_reward`)

`compute_reward(veh_id, delta_v)` is called only on **non-terminal** steps (after the SUMO tick).

It uses:
- current ego speed
- leader gap/leader speed and relative speed
- TLS distance and TLS state
- `delta_v` (for a smoothness penalty)

Components (in code order):

1) **Progress**
- `+0.1 * ego_speed`

2) **Safety distance penalties**
- `gap < 2.0m` ⇒ `-15`
- else if `gap < 2.5m` ⇒ `-6`

3) **Queue shaping** (only when leader is stopped: `leader_speed < 1.0`)
- Bad: stopped too far behind queue
  - `ego_speed < 0.5 and gap > 8.0` ⇒ `-12`
  - `ego_speed < 0.5 and gap > 5.0` ⇒ `-6`
- Good: “close but safe” stopping
  - `2.5 ≤ gap ≤ 4.0 and ego_speed < 0.7` ⇒ `+6`
- Good: creeping forward when far
  - `gap > 5.0 and 0.3 ≤ ego_speed ≤ 2.5` ⇒ `+2`

4) **Approaching leader too fast**
- `rel_speed > 2.0 and gap < 10.0` ⇒ `-4`
- `rel_speed > 4.0 and gap < 6.0` ⇒ `-8`

5) **Red light shaping** (only when red: `tls_red > 0.5`)
- Too fast close to red:
  - `tls_dist < 0.04 and ego_speed > 2.0` ⇒ `-8`
  - else if `tls_dist < 0.10 and ego_speed > 5.0` ⇒ `-4`
- Bad: stopped too far from red line
  - `ego_speed < 0.5 and tls_dist > 0.04 and gap > 6.0` ⇒ `-12`
- Good: creeping closer
  - `0.02 < tls_dist ≤ 0.08 and 0.3 ≤ ego_speed ≤ 2.5 and gap > 5.0` ⇒ `+2`
- Ideal stop zone (about 1.5–4m before TLS)
  - `0.0075 ≤ tls_dist ≤ 0.02 and ego_speed < 0.6` ⇒ `+8`
- Too close / crossing risk
  - `tls_dist < 0.0075 and ego_speed > 0.1` ⇒ `-15`

6) **Yellow caution**
- `tls_yellow > 0.5 and tls_dist < 0.10 and ego_speed > 6.0` ⇒ `-3`

7) **Unnecessary stop**
- if `ego_speed < 0.2` and `gap > 8.0` and not a “true red stop context” ⇒ `-6`
  - implemented as: `(tls_red < 0.5) or (tls_red > 0.5 and tls_dist > 0.04)`

8) **Smoothness penalty**
- `-0.05 * abs(delta_v)`

### 6.2 Terminal rewards (`utils.run_episode`)

Terminals are handled in `run_episode` and do **not** call `compute_reward` on that step:
- collision (ego in colliding list): `-30`, `end_reason="ego_crash"`
- arrived: `+20`, `end_reason="arrived"`
- abnormal disappearance: `-20`, `end_reason="abnormal_end"`
- timeout (after horizon): `-5`, `end_reason="timeout"`

---

## 7) Ego incident counting (correct metrics) — `ego_events.py`

Why: TraCI can report the same collision/teleport/emergency-stop across multiple steps.

`ego_events.accumulate_ego_events(...)` maintains per-episode state and returns **per-step deltas**.

Counters in `config.py`:
- `TOTAL_COLLISION_EVENTS`: unique collision events (system-wide)
- `TOTAL_EGO_COLLISIONS`: unique collision events involving ego
- `TOTAL_EGO_TELEPORTS`: ego teleports (max once per episode)
- `TOTAL_EGO_EMERGENCY_STOPS`: ego emergency stops (max once per episode)
- `TOTAL_EGO_CRASHES`: ego had any incident (collision/teleport/emergency) this episode (0/1 per episode)

De-dup key:
- uses `collision.collisionID` when available
- otherwise falls back to `(sim_time, collider, victim)`

---

## 8) Running a trained model (evaluation helper)

`utils.run_loaded_model_on_route(model_path, route_id, use_gui=False, max_steps=None)`:
- loads weights into a fresh `DQN(12, 5)`
- resets SUMO
- spawns ego and runs greedy actions (no epsilon)
- returns `(episode_reward, steps, end_reason)`

---

## 9) Common pitfalls / gotchas

- **Spawn return value is ignored in training**: `spawn_ego(...)` returns `(ok, reason)` but `run_episode` only checks `ego_exists()`.
- **One RL step = “apply speed then tick”**: action affects the *next* simulator tick.
- **Hybrid controller**: `apply_action` may override the agent’s intent near queues / red lights.
- **Epsilon restarts on new runs**: because `global_step` isn’t persisted by default.
- **Timeout transition uses `Action.KEEP`**: regardless of the last chosen action.

---

## 10) Quick “doc drift” checklist (when you change code)

If you touch environment/training code, re-check these items:
- `state_dim == len(get_state(...))` (currently 12)
- action ordering matches `Action(IntEnum)` indices
- any change to normalization constants (20 m/s, 100 m, 10 s, 200 m)
- terminal conditions still happen **after** `traci.simulationStep()`
- target net update condition uses **global_step**
- reward uses `delta_v` from `apply_action` (not actual acceleration)
