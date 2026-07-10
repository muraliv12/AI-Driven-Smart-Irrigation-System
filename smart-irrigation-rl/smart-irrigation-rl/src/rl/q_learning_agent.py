"""
q_learning_agent.py
--------------------
Tabular Q-Learning agent for discrete-state, discrete-action irrigation
control. Implements epsilon-greedy exploration with exponential decay,
the standard Q-Learning (off-policy TD control) update rule, and
save/load support for the learned Q-table.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from src.utils.config_loader import RLConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

State = Tuple[int, int, int, int]


class QLearningAgent:
    """
    Off-policy tabular Q-Learning agent.

    Update rule:
        Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

    Parameters
    ----------
    n_actions : int
        Number of discrete actions available (2 for this project).
    config : RLConfig
        Hyperparameters: learning rate, discount factor, epsilon schedule.
    """

    def __init__(self, n_actions: int, config: RLConfig):
        self.n_actions = n_actions
        self.cfg = config
        self.alpha = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay

        # Q-table: maps state tuple -> np.ndarray of shape (n_actions,)
        self.q_table: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))

        self._rng = np.random.default_rng(config.random_seed)

    def select_action(self, state: State, greedy: bool = False) -> int:
        """
        Choose an action via epsilon-greedy policy.

        Parameters
        ----------
        state : State
        greedy : bool
            If True, always exploit (used at evaluation/inference time).

        Returns
        -------
        int
            Selected action index.
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        q_values = self.q_table[state]
        # Break ties randomly rather than always picking the first max.
        max_q = np.max(q_values)
        candidates = np.flatnonzero(q_values == max_q)
        return int(self._rng.choice(candidates))

    def update(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        """Apply the Q-Learning TD update for a single transition."""
        current_q = self.q_table[state][action]
        future_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * future_q
        td_error = td_target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Exponentially decay exploration rate, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Persist the learned Q-table to disk via pickle."""
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        logger.info("Q-table saved to %s (%d states).", path, len(self.q_table))

    def load(self, path: str) -> None:
        """Load a previously trained Q-table from disk."""
        with open(path, "rb") as f:
            loaded: Dict[State, np.ndarray] = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), loaded)
        logger.info("Q-table loaded from %s (%d states).", path, len(self.q_table))
