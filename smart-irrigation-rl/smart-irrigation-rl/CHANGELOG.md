# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2025-01-20

### Added
- Initial public release of the AI-Driven Smart Irrigation System.
- `SensorSimulator` for offline generation of 7-signal synthetic IoT data
  (soil moisture, temperature, humidity, light intensity, rain, battery
  voltage, ambient pressure) with a coherent day/night cycle.
- `ArduinoSerialReader` for real-time data ingestion from an Arduino Uno
  over PySerial, matching the physical DHT11 / LDR / soil-moisture rig.
- `IrrigationEnv`: RL environment fusing continuous sensor data into a
  discretized 4-dimensional state space.
- `QLearningAgent`: tabular Q-Learning implementation with epsilon-greedy
  exploration, exponential epsilon decay, and Q-table persistence.
- Smart Irrigation Score (SIS) and symmetric reward function
  (`src/rl/reward.py`) penalizing both under- and over-watering.
- Rule-based single-threshold baseline controller for benchmarking.
- Matplotlib analytics dashboard: reward convergence, SIS trend, water-usage
  comparison, and per-episode irrigation timeline plots.
- CLI entrypoint (`src/main.py`) with `train`, `evaluate`, and `live`
  subcommands.
- Full unit and integration test suite (23 tests) covering sensors, reward
  shaping, environment dynamics, the Q-Learning agent, and an end-to-end
  training run.
- YAML-based configuration system (`config/config.yaml`) with typed
  dataclass loading and validation.
- Sample synthetic sensor dataset (`data/sample_sensor_data.csv`).
- Complete project documentation: README, architecture diagrams, module
  reference, and project report.

### Results
- Trained agent achieves a **33.9% reduction in water usage** versus the
  rule-based baseline over 100 evaluation episodes, while improving the
  average Smart Irrigation Score from 79.3 to 86.6.

## [Unreleased]

### Planned
- Deep Q-Network (DQN) variant for continuous state representation.
- Multi-zone irrigation support (independent Q-tables per field zone).
- REST API wrapper for remote dashboard access.
