# Module Reference

Complete reference for every module, class, and function in the project.
Also see [ARCHITECTURE.md](ARCHITECTURE.md) for system-level diagrams.

---

## `src/sensors/sensor_simulator.py`

### `SensorReading` (dataclass)
Unified 7-field sensor state vector consumed everywhere downstream
(environment, reward, dashboard).

| Field | Type | Description |
|---|---|---|
| `timestamp_step` | `int` | Discrete simulation/serial step index. |
| `soil_moisture` | `float` | Percentage, 0–100. |
| `temperature` | `float` | Degrees Celsius (DHT11). |
| `humidity` | `float` | Percentage relative humidity (DHT11). |
| `light_intensity` | `float` | LDR analog reading, 0–1023. |
| `rain_detected` | `bool` | Digital rain-sensor state. |
| `battery_voltage` | `float` | Volts, solar/battery monitor. |
| `ambient_pressure` | `float` | hPa, barometric sensor. |

`to_dict() -> dict` — flattens the dataclass for CSV logging / JSON export.

### `SensorSimulator`
Generates temporally-coherent synthetic sensor data with a day/night cycle.

- `__init__(seed: int = 42, steps_per_day: int = 96)`
- `reset() -> None` — resets internal soil-moisture state to 55%.
- `read(step: int, irrigated_last_step: bool = False) -> SensorReading` —
  advances the simulated environment by one step; soil moisture depletes
  via a day-phase-dependent evapotranspiration term and rises when
  irrigation or rain occurred on the previous step.

**Internal helper**: `_day_phase(step) -> float` returns a 0–1 sinusoid
representing the position in the day/night cycle (peaks at simulated
midday), driving temperature, humidity, and light generation.

---

## `src/sensors/serial_reader.py`

### `ArduinoSerialReader`
PySerial wrapper that reads CSV payloads
(`soil_moisture,temperature,humidity,light,rain,battery,pressure`) streamed
by the Arduino Uno firmware and converts them into `SensorReading` objects.

- `connect() / disconnect()` — open/close the serial port; `connect()`
  sleeps 2s after opening to allow the Arduino's auto-reset to complete.
- `read(retries: int = 3) -> SensorReading` — reads one line, retries on
  malformed/empty payloads, raises `SerialReadError` after exhausting
  retries.
- Context-manager support (`with ArduinoSerialReader(...) as reader:`).

### `SerialReadError`
Raised when a line cannot be parsed into a valid 7-field `SensorReading`.

---

## `src/rl/reward.py`

### `compute_sis(reading, weights) -> float`
Computes the **Smart Irrigation Score** (0–100): a weighted composite of
band-scores for soil moisture (55% weight), temperature (20%), humidity
(15%), and light (10%). Each factor uses `_band_score`, which returns 1.0
inside an ideal range and decays linearly to 0 outside it.

### `symmetric_reward(reading, action, water_cost_per_action, ideal_moisture, tolerance) -> float`
The RL reward signal, combining:
1. A **symmetric quadratic moisture-deviation penalty** — equal cost for
   being N% too dry or N% too wet relative to the 55% ideal.
2. The normalized SIS as a positive shaping term.
3. A fixed **water-cost penalty** whenever the agent irrigates.
4. An extra penalty for irrigating while rain is detected (wasteful).

---

## `src/rl/environment.py`

### `IrrigationEnv`
Gym-like environment (no external Gym dependency) wrapping the sensor
simulator and exposing `reset()` / `step(action)`.

- **Actions**: `0` = do not irrigate, `1` = irrigate.
- **State**: 4-tuple `(moisture_bin, temp_bin, humidity_bin, light_bin)`,
  produced by `_discretize()` via `numpy.digitize` against bin edges
  computed from `config.rl.n_*_bins` and `config.thresholds`.
- `reset() -> State` — resets the simulator and returns the initial state.
- `step(action) -> (next_state, reward, done, info)` — advances one
  simulated step; `info` contains the raw reading dict, SIS score, step
  index, and a `water_used` flag for metrics.

---

## `src/rl/q_learning_agent.py`

### `QLearningAgent`
Tabular Q-Learning with epsilon-greedy exploration.

- `select_action(state, greedy=False) -> int` — epsilon-greedy action
  selection; ties among equally-good actions are broken randomly.
- `update(state, action, reward, next_state, done) -> None` — applies the
  standard off-policy TD update:
  `Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`, with the
  future term zeroed out when `done=True`.
- `decay_epsilon() -> None` — multiplies epsilon by `epsilon_decay`,
  floored at `epsilon_min`.
- `save(path) / load(path)` — pickle persistence of the Q-table
  (`dict[state] -> np.ndarray` of shape `(n_actions,)`).

---

## `src/rl/train.py`

### `train_agent(config, checkpoint_every=100) -> (agent, env, history)`
Full training loop: iterates `config.rl.n_episodes` episodes, each running
up to `config.rl.max_steps_per_episode` steps, applying epsilon-greedy
action selection and Q-Learning updates at every step, decaying epsilon
once per episode, logging `EpisodeMetrics`, and periodically checkpointing
the Q-table to `config.paths.q_table_file`.

---

## `src/rl/evaluate.py`

### `evaluate_agent(agent, env, n_episodes=50) -> TrainingHistory`
Runs the trained agent in **greedy mode** (`epsilon` ignored) for
`n_episodes`, collecting the same metrics as training for fair comparison.

### `compare_to_baseline(config, agent, n_episodes=50) -> (summary, rl_history, baseline_history)`
Runs both the trained agent and the rule-based baseline
(`metrics.run_rule_based_baseline`) on independently-seeded environments
and computes the percentage water-usage reduction and average SIS for
each.

### `run_single_episode_timeline(agent, env) -> (steps, soil_moisture, actions)`
Runs one greedy evaluation episode and records the full soil-moisture /
action trajectory, used by `dashboard.plot_irrigation_timeline`.

---

## `src/analytics/metrics.py`

### `EpisodeMetrics` (dataclass)
Per-episode summary: `episode`, `total_reward`, `water_used`, `avg_sis`,
`epsilon`.

### `TrainingHistory`
Accumulates `EpisodeMetrics` across a run; `to_csv(path)` persists the
full history, `rewards()` / `water_usage()` / `avg_sis_series()` return
flat lists for plotting.

### `rule_based_action(reading, low_threshold=52.0) -> int`
The naive single-threshold baseline controller: irrigate whenever soil
moisture falls below a fixed threshold, ignoring every other sensor
signal (temperature, humidity, light, rain) — representative of
non-AI timer/threshold irrigation controllers.

### `run_rule_based_baseline(env, n_episodes) -> TrainingHistory`
Runs the rule-based policy through the same `IrrigationEnv` used for RL
training/evaluation, producing directly comparable metrics.

---

## `src/analytics/dashboard.py`

All functions save a PNG to the given `output_path` using a headless
(`Agg`) Matplotlib backend, safe for CI/servers without a display.

- `plot_reward_convergence(history, output_path, window=20)` — raw +
  moving-average reward per episode.
- `plot_water_usage_comparison(rl_history, baseline_history, output_path)`
  — bar chart of average irrigation events/episode with the computed
  percentage reduction in the title.
- `plot_sis_trend(history, output_path, window=20)` — SIS trend over
  training with moving average.
- `plot_irrigation_timeline(steps, soil_moisture, actions, output_path)` —
  soil-moisture curve for a single evaluation episode with irrigation
  events marked and the ideal-moisture band shaded.

---

## `src/utils/config_loader.py`

Typed dataclasses (`SerialConfig`, `SensorThresholds`, `RLConfig`,
`PathsConfig`, `AppConfig`) plus `load_config(path) -> AppConfig`, which
parses `config/config.yaml`, ignoring unknown keys and raising
`ConfigError` on missing files or invalid YAML.

## `src/utils/logger.py`

`get_logger(name, log_level="INFO") -> logging.Logger` — configures a
shared root logger (console + rotating file handler under
`outputs/logs/`) on first call and returns a namespaced logger on every
call thereafter.

---

## `src/main.py`

CLI entrypoint built with `argparse`.

| Command | Description |
|---|---|
| `python -m src.main train` | Train a new Q-Learning agent, save the Q-table, and generate reward/SIS plots. |
| `python -m src.main evaluate --episodes N` | Load the saved Q-table, compare against the rule-based baseline, and generate water-usage/timeline plots. |
| `python -m src.main live --port COM3` | Stream live Arduino sensor data and print the trained agent's real-time irrigation decisions. |

Global `--config` flag overrides the default `config/config.yaml` path for
all subcommands.
