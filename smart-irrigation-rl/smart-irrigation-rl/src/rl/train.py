"""
train.py
--------
Orchestrates the full Q-Learning training loop: episode iteration,
epsilon-greedy action selection, environment stepping, TD updates,
per-episode metric logging, and periodic checkpointing of the Q-table.
"""

from __future__ import annotations

from src.analytics.metrics import EpisodeMetrics, TrainingHistory
from src.rl.environment import IrrigationEnv
from src.rl.q_learning_agent import QLearningAgent
from src.utils.config_loader import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_agent(config: AppConfig, checkpoint_every: int = 100) -> tuple[QLearningAgent, IrrigationEnv, TrainingHistory]:
    """
    Train a Q-Learning agent on the IrrigationEnv for `config.rl.n_episodes`
    episodes.

    Parameters
    ----------
    config : AppConfig
        Fully loaded application configuration.
    checkpoint_every : int
        Save the Q-table to disk every N episodes (in addition to at the end).

    Returns
    -------
    agent : QLearningAgent
        The trained agent (with its final Q-table).
    env : IrrigationEnv
        The environment instance used for training (reusable for evaluation).
    history : TrainingHistory
        Per-episode metrics collected during training.
    """
    env = IrrigationEnv(config.rl, config.thresholds, seed=config.rl.random_seed)
    agent = QLearningAgent(n_actions=env.N_ACTIONS, config=config.rl)
    history = TrainingHistory()

    logger.info(
        "Starting training: %d episodes, %d steps/episode, state space %s",
        config.rl.n_episodes, config.rl.max_steps_per_episode, env.state_space_size,
    )

    for episode in range(1, config.rl.n_episodes + 1):
        state = env.reset()
        total_reward, water_used, sis_values = 0.0, 0, []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            water_used += info["water_used"]
            sis_values.append(info["sis"])

        agent.decay_epsilon()
        avg_sis = sum(sis_values) / len(sis_values) if sis_values else 0.0

        history.add(EpisodeMetrics(
            episode=episode,
            total_reward=round(total_reward, 3),
            water_used=water_used,
            avg_sis=round(avg_sis, 2),
            epsilon=round(agent.epsilon, 4),
        ))

        if episode % 50 == 0 or episode == config.rl.n_episodes:
            logger.info(
                "Episode %4d/%d | reward=%.2f | water_used=%d | avg_sis=%.1f | epsilon=%.3f",
                episode, config.rl.n_episodes, total_reward, water_used, avg_sis, agent.epsilon,
            )

        if checkpoint_every and episode % checkpoint_every == 0:
            agent.save(config.paths.q_table_file)

    agent.save(config.paths.q_table_file)
    history.to_csv(config.paths.training_log_file)
    logger.info("Training complete. Final Q-table has %d states.", len(agent.q_table))

    return agent, env, history
