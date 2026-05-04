````markdown
# rl-traffic-agent

RL training of an ego vehicle in **SUMO** using **TraCI**.  
Implemented algorithms:

- **DQN**
- **PPO**

Both agents use the same:

- **12D state**
- **5 discrete speed actions**
- shared reward and terminal conditions

This allows direct comparison of safety, trip completion, and robustness in traffic. :contentReference[oaicite:0]{index=0}

## Run

### DQN
```bash
python train.py
python validate.py
````

### PPO

```bash
python ppo/train_ppo.py
python validate_ppo.py
```

## Main files

* `sumo_utils.py` — environment, state, reward, action application
* `Action.py` — discrete actions
* `train.py`, `utils.py` — DQN
* `ppo/train_ppo.py`, `ppo/PPO_actor_critic.py` — PPO
* `validate.py`, `validate_ppo.py` — evaluation
* `ego_events.py`, `logger_utils.py` — metrics and logs

## Notes

* SUMO is controlled through `TraCI`
* one episode = reset SUMO, spawn ego, run until arrival/crash/timeout
* validation runs on separate routes and traffic scales
* logs are written to `logs/`

```
```
