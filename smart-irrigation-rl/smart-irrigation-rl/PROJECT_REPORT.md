# Project Report: AI-Driven Smart Irrigation System

**Author:** Khuti
**Duration:** October 2024 – January 2025
**Domain:** Reinforcement Learning · IoT · Precision Agriculture

---

## Abstract

Water scarcity and inefficient agricultural water use are pressing
challenges in precision agriculture. This project presents an AI-driven
smart irrigation controller that fuses readings from 7+ IoT sensors —
soil moisture, temperature, humidity, light, rain, battery, and
barometric pressure — collected via an Arduino Uno, into a unified state
representation for a reinforcement-learning (RL) agent. A tabular
Q-Learning agent is trained against a custom reward function, the
**Smart Irrigation Score (SIS)**, combined with a symmetric moisture-
deviation penalty and an explicit water-usage cost. Trained over 600
episodes, the agent achieves a **33.9% reduction in water usage** compared
to a naive rule-based threshold controller, while simultaneously
*improving* the average Smart Irrigation Score from 79.3 to 86.6,
demonstrating that sensor fusion and learned control can reduce waste
without compromising crop-health outcomes.

## Introduction

Irrigation is one of the largest consumers of freshwater in agriculture.
Conventional controllers typically rely on a fixed timer or a single
moisture threshold, disregarding the compounding effects of temperature,
humidity, ambient light, and precipitation. This project investigates
whether a reinforcement-learning agent, trained on fused multi-sensor
data, can learn an irrigation policy that is both more water-efficient
and better at maintaining favorable growing conditions than such rule-
based approaches.

## Problem Statement

Given a stream of real-time sensor readings describing field conditions,
determine, at each time step, whether to irrigate or withhold irrigation,
such that:
1. Soil moisture is kept within an ideal range for plant health.
2. Total water consumption is minimized.
3. The decision accounts for all available sensor signals jointly,
   rather than a single threshold in isolation.

## Objectives

- Design a sensor-fusion pipeline that converts raw multi-sensor IoT data
  into a compact state representation suitable for tabular RL.
- Define a reward function (the Smart Irrigation Score and its associated
  symmetric penalty) that captures the trade-off between crop health and
  water conservation.
- Train a Q-Learning agent to convergence and benchmark it against a
  naive rule-based baseline.
- Build an analytics layer to visualize training convergence and
  quantify water savings.
- Package the system so it can run entirely offline (simulation) or be
  connected to physical Arduino Uno hardware without any code changes to
  the RL core.

## Literature Review

Reinforcement learning has been increasingly applied to resource-
constrained control problems where the optimal policy is difficult to
hand-specify but can be learned from interaction with an environment.
Classical **Q-Learning** (Watkins, 1989) is a model-free, off-policy
temporal-difference algorithm well suited to small, discretizable state-
action spaces such as this one, where the full state space (240 discrete
states) is small enough for a tabular representation to converge quickly
without the sample-inefficiency concerns that motivate function-
approximation approaches (e.g., Deep Q-Networks) in larger domains.
Precision-agriculture research has separately established that
multi-sensor fusion (soil, atmospheric, and light sensors together)
outperforms single-sensor threshold rules for irrigation scheduling,
motivating the sensor-fusion state design used here.

## Methodology

### 1. Sensor Fusion
Seven sensor channels are fused into a single `SensorReading` state
vector at every time step: soil moisture, temperature, humidity, light
intensity, rain detection, battery voltage, and barometric pressure.

### 2. State Discretization
The four control-relevant continuous signals (soil moisture, temperature,
humidity, light) are discretized into 5, 4, 4, and 3 bins respectively
via `numpy.digitize` against configurable bin edges, producing a
4-tuple state with a maximum of 240 possible combinations.

### 3. Reward Design — Smart Irrigation Score (SIS)
The SIS is a weighted composite (moisture 55%, temperature 20%, humidity
15%, light 10%) of per-factor "band scores," each equal to 1.0 inside an
ideal range and decaying linearly to 0 outside it. The **symmetric
reward function** combines:
- A quadratic penalty on soil-moisture deviation from the 55% ideal,
  identical in magnitude whether the deviation is above or below target.
- The normalized SIS as a positive shaping term.
- A fixed water-cost penalty applied whenever the agent irrigates.
- An additional penalty for irrigating while rain is currently detected.

### 4. Agent Training
A tabular `QLearningAgent` is trained for 600 episodes of 96 steps each
(representing a 15-minute-resolution simulated day), using epsilon-greedy
exploration annealed from 1.0 to 0.05 at a 0.99 per-episode decay rate,
learning rate 0.15, and discount factor 0.95.

### 5. Baseline Comparison
A rule-based controller irrigating whenever soil moisture falls below a
fixed 52% threshold — ignoring all other signals — serves as the
baseline, representative of commodity timer/threshold irrigation
controllers.

### 6. Evaluation
Both policies are run for 100 independent evaluation episodes (with the
trained agent in pure-exploitation/greedy mode) on separately-seeded
environment instances, and average irrigation-event counts and SIS are
compared.

## System Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full Mermaid
diagrams (system architecture, data flow, training workflow, class
diagram, sequence diagram, and deployment diagram). At a high level:

```
Arduino Uno (sensors) --serial--> ArduinoSerialReader --\
                                                          --> IrrigationEnv --> QLearningAgent --> action
SensorSimulator (offline) ------------------------------/                          |
                                                                                     v
                                                                        Reward (SIS + symmetric penalty)
```

## Algorithms

**Q-Learning update rule:**

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

where `α` = 0.15 (learning rate), `γ` = 0.95 (discount factor), `r` is
the symmetric reward, `s` the current discretized state, and `a` the
chosen action (irrigate / hold).

**Epsilon-greedy exploration:**

```
a = random action        with probability ε
a = argmax_a Q(s, a)     with probability 1 − ε
ε ← max(ε_min, ε · decay)   (applied once per episode)
```

## Technologies Used

- **Python 3.10+** — core implementation language.
- **NumPy** — Q-table storage, state discretization, vectorized reward
  computation.
- **Matplotlib** — reward convergence, SIS trend, water-usage, and
  irrigation-timeline visualizations.
- **PyYAML** — externalized, typed configuration.
- **PySerial** — real-time ingestion from the Arduino Uno over USB
  serial.
- **Arduino Uno + DHT11 / LDR / capacitive soil-moisture sensor** —
  physical sensing hardware.
- **Pytest** — automated unit and integration testing (23 tests).

## Implementation

The codebase is organized into four packages under `src/`:
`sensors` (data acquisition, real and simulated), `rl` (environment,
reward, agent, training, evaluation), `analytics` (metrics tracking,
rule-based baseline, dashboard plotting), and `utils` (configuration
loading, logging). A single CLI entrypoint (`src/main.py`) exposes
`train`, `evaluate`, and `live` subcommands. All hyperparameters and file
paths are externalized to `config/config.yaml` and loaded into typed
dataclasses, avoiding hardcoded magic numbers throughout the codebase.

## Results

| Metric | Rule-Based Baseline | Q-Learning Agent | Change |
|---|---|---|---|
| Avg. irrigation events / episode (100 eval episodes) | 1.89 | 1.25 | **−33.9%** |
| Avg. Smart Irrigation Score | 79.3 | 86.6 | **+7.3 pts** |
| Training episodes to convergence | — | ~450–600 | — |
| Reachable states (of 240 possible) | — | 50 | — |

The reward-convergence and SIS-trend plots (`outputs/reward_convergence.png`,
`outputs/sis_trend.png`) show the expected pattern of high-variance early
episodes (driven by epsilon-greedy exploration) settling into a
consistently higher-reward regime as epsilon anneals toward its floor.
The water-usage comparison plot (`outputs/water_usage_comparison.png`)
visualizes the 33.9% reduction directly, and the irrigation-timeline plot
(`outputs/irrigation_timeline.png`) illustrates the trained agent's
tendency to irrigate only when soil moisture approaches the lower edge of
the ideal band, rather than reactively at a fixed threshold.

## Advantages

- **Water efficiency**: ~34% fewer irrigation events than a naive
  threshold controller, directly reducing water consumption.
- **Better growing conditions**: higher average SIS despite using less
  water, because decisions account for all sensor signals jointly.
- **Interpretability**: the tabular Q-table (≈50 learned states) can be
  inspected directly, unlike a black-box neural policy.
- **Hardware-agnostic core**: the same trained policy runs against either
  simulated or live Arduino sensor data with no code changes to the RL
  logic.
- **Configurable and testable**: all thresholds and hyperparameters are
  externalized; 23 automated tests validate every component.

## Limitations

- The tabular representation does not scale to finer-grained
  discretization or additional sensor channels without an exponential
  growth in state-space size.
- The simulator's sensor dynamics, while designed to be realistic, are
  synthetic; real-world validation against a physical field deployment
  over a full growing season has not yet been performed.
- The rule-based baseline, while representative of common commodity
  controllers, is intentionally simple; more sophisticated non-RL
  baselines (e.g., PID controllers, evapotranspiration models) were not
  benchmarked.
- The current reward function does not account for crop-specific
  moisture requirements (e.g., different ideal bands per crop type).

## Future Scope

- Migrate to a Deep Q-Network (DQN) or actor-critic method to support a
  continuous state space and avoid discretization-driven information
  loss.
- Incorporate live weather-forecast data to enable proactive (rather than
  reactive) irrigation decisions ahead of predicted rainfall.
- Extend to multi-zone irrigation with per-zone policies sharing a
  common weather context.
- Deploy on-device (TinyML on the Arduino itself, or a Raspberry Pi) to
  remove the dependency on a companion laptop for live inference.
- Conduct a field trial comparing real crop-health and water-usage
  outcomes against the simulated results reported here.

## Conclusion

This project demonstrates that a relatively simple tabular Q-Learning
agent, trained on a well-designed sensor-fusion state representation and
a carefully shaped symmetric reward function, can substantially reduce
water usage (33.9%) relative to a naive rule-based irrigation controller
while simultaneously improving average growing-condition quality (SIS
+7.3 points). The system is fully reproducible offline via a physically-
grounded sensor simulator and can be deployed against real Arduino Uno
hardware without modification to the learned policy or RL core,
demonstrating a practical, low-cost path from simulation to field
deployment for smart irrigation.

## References

1. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD
   Thesis, Cambridge University.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
   Introduction* (2nd ed.). MIT Press.
3. Arduino Uno DHT11 / LDR / Capacitive Soil Moisture Sensor
   documentation (manufacturer datasheets).
4. Project source code and full documentation:
   `README.md`, `docs/ARCHITECTURE.md`, `docs/MODULES.md`.
