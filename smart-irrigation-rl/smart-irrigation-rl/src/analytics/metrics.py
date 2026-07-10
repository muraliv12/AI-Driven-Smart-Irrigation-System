"""
metrics.py
----------
Tracks per-episode training metrics (reward convergence, water usage,
average SIS) and implements the simple rule-based irrigation baseline
that the RL agent is benchmarked against (the "34% reduction in water
usage vs. rule-based baselines" figure quoted in the project summary).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List

from src.rl.environment import IrrigationEnv
from src.sensors.sensor_simulator import SensorReading


@dataclass
class EpisodeMetrics:
    episode: int
    total_reward: float
    water_used: int
    avg_sis: float
    epsilon: float


@dataclass
class TrainingHistory:
    """Accumulates EpisodeMetrics across a full training run."""

    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add(self, metrics: EpisodeMetrics) -> None:
        self.episodes.append(metrics)

    def to_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_reward", "water_used", "avg_sis", "epsilon"])
            for m in self.episodes:
                writer.writerow([m.episode, m.total_reward, m.water_used, m.avg_sis, m.epsilon])

    def rewards(self) -> List[float]:
        return [m.total_reward for m in self.episodes]

    def water_usage(self) -> List[int]:
        return [m.water_used for m in self.episodes]

    def avg_sis_series(self) -> List[float]:
        return [m.avg_sis for m in self.episodes]


def rule_based_action(reading: SensorReading, low_threshold: float = 52.0) -> int:
    """
    Simple rule-based irrigation baseline: irrigate whenever soil moisture
    drops below a single fixed threshold, regardless of temperature,
    humidity, light, or current rain. This mirrors the naive timer/threshold
    controllers commonly deployed in non-AI irrigation setups -- it only
    looks at one signal and errs conservatively (i.e. "safe" but wasteful),
    which is exactly the inefficiency the RL agent is trained to remove by
    fusing all 7+ sensor signals into its policy.

    Parameters
    ----------
    reading : SensorReading
    low_threshold : float
        Soil moisture percentage below which irrigation is triggered.

    Returns
    -------
    int
        0 (no irrigation) or 1 (irrigate).
    """
    return 1 if reading.soil_moisture < low_threshold else 0


def run_rule_based_baseline(env: IrrigationEnv, n_episodes: int) -> TrainingHistory:
    """
    Run the rule-based baseline policy through the same environment used
    for RL training, recording identical metrics for a fair comparison.

    Parameters
    ----------
    env : IrrigationEnv
    n_episodes : int

    Returns
    -------
    TrainingHistory
    """
    history = TrainingHistory()
    for ep in range(1, n_episodes + 1):
        env.reset()
        total_reward, water_used, sis_values = 0.0, 0, []
        done = False
        reading = env._last_reading  # after reset(), holds the initial reading
        while not done:
            action = rule_based_action(reading)
            _, reward, done, info = env.step(action)
            total_reward += reward
            water_used += info["water_used"]
            sis_values.append(info["sis"])
            reading = env._last_reading
        avg_sis = sum(sis_values) / len(sis_values) if sis_values else 0.0
        history.add(EpisodeMetrics(ep, round(total_reward, 3), water_used, round(avg_sis, 2), 0.0))
    return history
