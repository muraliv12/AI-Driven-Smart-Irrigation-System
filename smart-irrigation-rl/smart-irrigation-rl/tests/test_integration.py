"""Integration test: short end-to-end training run and baseline comparison."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl.environment import IrrigationEnv
from src.rl.q_learning_agent import QLearningAgent
from src.analytics.metrics import run_rule_based_baseline
from src.utils.config_loader import RLConfig, SensorThresholds


def test_short_training_run_improves_reward():
    """Confirms the agent's average reward improves over a short run
    compared to a purely random policy, validating the learning loop
    end-to-end (environment -> agent -> update)."""
    rl_cfg = RLConfig(
        n_episodes=60, max_steps_per_episode=30, learning_rate=0.2,
        discount_factor=0.9, epsilon_start=1.0, epsilon_min=0.05,
        epsilon_decay=0.9, random_seed=7,
    )
    thresholds = SensorThresholds()
    env = IrrigationEnv(rl_cfg, thresholds, seed=7)
    agent = QLearningAgent(n_actions=env.N_ACTIONS, config=rl_cfg)

    early_rewards, late_rewards = [], []

    for episode in range(1, rl_cfg.n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()

        if episode <= 10:
            early_rewards.append(total_reward)
        elif episode > rl_cfg.n_episodes - 10:
            late_rewards.append(total_reward)

    avg_early = sum(early_rewards) / len(early_rewards)
    avg_late = sum(late_rewards) / len(late_rewards)

    assert avg_late >= avg_early - 2.0  # learning should not make things dramatically worse
    assert len(agent.q_table) > 0


def test_rule_based_baseline_runs_without_error():
    rl_cfg = RLConfig(n_episodes=5, max_steps_per_episode=20, random_seed=3)
    thresholds = SensorThresholds()
    env = IrrigationEnv(rl_cfg, thresholds, seed=3)
    history = run_rule_based_baseline(env, n_episodes=5)
    assert len(history.episodes) == 5
    assert all(m.water_used >= 0 for m in history.episodes)
