"""
evaluate.py
-----------
Evaluates a trained Q-Learning agent in greedy (exploitation-only) mode,
runs the rule-based baseline for comparison, and produces a single-episode
irrigation timeline used by the dashboard.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.analytics.metrics import TrainingHistory, EpisodeMetrics, run_rule_based_baseline
from src.rl.environment import IrrigationEnv
from src.rl.q_learning_agent import QLearningAgent
from src.utils.config_loader import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_agent(agent: QLearningAgent, env: IrrigationEnv, n_episodes: int = 50) -> TrainingHistory:
    """
    Run the trained agent greedily (no exploration) for `n_episodes` and
    collect the same metrics used during training, for apples-to-apples
    comparison against the rule-based baseline.
    """
    history = TrainingHistory()
    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward, water_used, sis_values = 0.0, 0, []
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            water_used += info["water_used"]
            sis_values.append(info["sis"])
        avg_sis = sum(sis_values) / len(sis_values) if sis_values else 0.0
        history.add(EpisodeMetrics(ep, round(total_reward, 3), water_used, round(avg_sis, 2), 0.0))
    return history


def compare_to_baseline(config: AppConfig, agent: QLearningAgent, n_episodes: int = 50) -> Dict[str, float]:
    """
    Compare the trained agent's average water usage against the rule-based
    baseline over `n_episodes` evaluation episodes, returning summary stats
    including percentage water-usage reduction.
    """
    eval_env = IrrigationEnv(config.rl, config.thresholds, seed=config.rl.random_seed + 1)
    baseline_env = IrrigationEnv(config.rl, config.thresholds, seed=config.rl.random_seed + 1)

    rl_history = evaluate_agent(agent, eval_env, n_episodes)
    baseline_history = run_rule_based_baseline(baseline_env, n_episodes)

    rl_water = rl_history.water_usage()
    baseline_water = baseline_history.water_usage()
    rl_avg = sum(rl_water) / len(rl_water)
    baseline_avg = sum(baseline_water) / len(baseline_water)
    reduction_pct = 100 * (baseline_avg - rl_avg) / baseline_avg if baseline_avg else 0.0

    rl_avg_sis = sum(rl_history.avg_sis_series()) / n_episodes
    baseline_avg_sis = sum(baseline_history.avg_sis_series()) / n_episodes

    summary = {
        "rl_avg_water_events": round(rl_avg, 2),
        "baseline_avg_water_events": round(baseline_avg, 2),
        "water_reduction_pct": round(reduction_pct, 2),
        "rl_avg_sis": round(rl_avg_sis, 2),
        "baseline_avg_sis": round(baseline_avg_sis, 2),
    }
    logger.info("Evaluation summary: %s", summary)
    return summary, rl_history, baseline_history


def run_single_episode_timeline(agent: QLearningAgent, env: IrrigationEnv) -> Tuple[List[int], List[float], List[int]]:
    """
    Run one greedy evaluation episode and record the soil-moisture /
    irrigation-action timeline for visualization.

    Returns
    -------
    steps, soil_moisture_series, action_series
    """
    steps: List[int] = []
    soil_moisture: List[float] = []
    actions: List[int] = []

    state = env.reset()
    steps.append(0)
    soil_moisture.append(env._last_reading.soil_moisture)
    actions.append(0)

    done = False
    while not done:
        action = agent.select_action(state, greedy=True)
        state, _, done, info = env.step(action)
        steps.append(info["step"])
        soil_moisture.append(info["reading"]["soil_moisture"])
        actions.append(action)

    return steps, soil_moisture, actions
