"""
dashboard.py
------------
Matplotlib-based analytics dashboard that visualizes:
  1. Reward convergence over training episodes (RL vs. moving average).
  2. Water usage per episode (RL agent vs. rule-based baseline).
  3. Average Smart Irrigation Score (SIS) trend across training.
  4. Irrigation event timeline for a single evaluation episode.

All plots are saved as PNG files under `outputs/` so they can be embedded
directly into the README / project report.
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np

from src.analytics.metrics import TrainingHistory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _moving_average(values: List[float], window: int = 20) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_reward_convergence(history: TrainingHistory, output_path: str, window: int = 20) -> None:
    """Plot per-episode reward and its moving average, save to output_path."""
    rewards = history.rewards()
    ma = _moving_average(rewards, window)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, label="Episode reward", color="#4C72B0")
    if len(ma) > 0:
        offset = len(rewards) - len(ma)
        plt.plot(range(offset, len(rewards)), ma, color="#DD8452", linewidth=2,
                  label=f"{window}-episode moving average")
    plt.title("Q-Learning Reward Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(output_path)


def plot_water_usage_comparison(
    rl_history: TrainingHistory, baseline_history: TrainingHistory, output_path: str
) -> None:
    """Bar/line comparison of water usage: RL agent vs. rule-based baseline."""
    rl_water = np.array(rl_history.water_usage())
    baseline_water = np.array(baseline_history.water_usage())

    rl_avg = rl_water.mean() if len(rl_water) else 0
    baseline_avg = baseline_water.mean() if len(baseline_water) else 0
    reduction_pct = (
        100 * (baseline_avg - rl_avg) / baseline_avg if baseline_avg > 0 else 0.0
    )

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Rule-Based Baseline", "Q-Learning Agent"],
        [baseline_avg, rl_avg],
        color=["#C44E52", "#55A868"],
    )
    plt.ylabel("Avg. Irrigation Events per Episode")
    plt.title(f"Water Usage Comparison ({reduction_pct:.1f}% reduction)")
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}", (bar.get_x() + bar.get_width() / 2, height),
                     textcoords="offset points", xytext=(0, 4), ha="center")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(output_path)
    logger.info("Water usage reduction vs. baseline: %.1f%%", reduction_pct)


def plot_sis_trend(history: TrainingHistory, output_path: str, window: int = 20) -> None:
    """Plot the average Smart Irrigation Score trend across training episodes."""
    sis_values = history.avg_sis_series()
    ma = _moving_average(sis_values, window)

    plt.figure(figsize=(10, 5))
    plt.plot(sis_values, alpha=0.35, color="#8172B2", label="Avg. SIS per episode")
    if len(ma) > 0:
        offset = len(sis_values) - len(ma)
        plt.plot(range(offset, len(sis_values)), ma, color="#937860", linewidth=2,
                  label=f"{window}-episode moving average")
    plt.title("Smart Irrigation Score (SIS) Trend")
    plt.xlabel("Episode")
    plt.ylabel("Average SIS (0-100)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(output_path)


def plot_irrigation_timeline(
    steps: List[int], soil_moisture: List[float], actions: List[int], output_path: str
) -> None:
    """
    Plot soil moisture over one evaluation episode with irrigation events
    marked, illustrating when and why the trained agent chose to irrigate.
    """
    plt.figure(figsize=(11, 5))
    plt.plot(steps, soil_moisture, color="#4C72B0", label="Soil Moisture (%)")
    irrigate_steps = [s for s, a in zip(steps, actions) if a == 1]
    irrigate_moisture = [m for m, a in zip(soil_moisture, actions) if a == 1]
    plt.scatter(irrigate_steps, irrigate_moisture, color="#DD8452", zorder=5,
                label="Irrigation Event", marker="v", s=60)
    plt.axhspan(45, 65, color="green", alpha=0.08, label="Ideal Moisture Band")
    plt.title("Irrigation Event Timeline (Evaluation Episode)")
    plt.xlabel("Time Step")
    plt.ylabel("Soil Moisture (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(output_path)


def _save(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved plot to %s", output_path)
