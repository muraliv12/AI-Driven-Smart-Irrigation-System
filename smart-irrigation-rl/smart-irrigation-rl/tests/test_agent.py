"""Unit tests for src.rl.q_learning_agent."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl.q_learning_agent import QLearningAgent
from src.utils.config_loader import RLConfig


def _make_agent(seed=1):
    cfg = RLConfig(random_seed=seed, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9)
    return QLearningAgent(n_actions=2, config=cfg)


def test_new_state_initializes_to_zero_q_values():
    agent = _make_agent()
    q_values = agent.q_table[(0, 0, 0, 0)]
    assert np.array_equal(q_values, np.zeros(2))


def test_update_moves_q_value_toward_target():
    agent = _make_agent()
    state, next_state = (0, 0, 0, 0), (1, 0, 0, 0)
    agent.update(state, action=1, reward=1.0, next_state=next_state, done=False)
    assert agent.q_table[state][1] > 0.0


def test_epsilon_decays_and_is_floored():
    agent = _make_agent()
    for _ in range(200):
        agent.decay_epsilon()
    assert agent.epsilon == agent.epsilon_min


def test_select_action_greedy_picks_max_q():
    agent = _make_agent()
    state = (2, 1, 1, 1)
    agent.q_table[state] = np.array([0.1, 5.0])
    action = agent.select_action(state, greedy=True)
    assert action == 1


def test_save_and_load_roundtrip():
    agent = _make_agent()
    state = (3, 2, 1, 0)
    agent.q_table[state] = np.array([0.5, 1.5])

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "q_table.pkl")
        agent.save(path)

        new_agent = _make_agent()
        new_agent.load(path)
        assert np.array_equal(new_agent.q_table[state], np.array([0.5, 1.5]))


def test_terminal_state_has_no_future_reward_contribution():
    agent = _make_agent()
    state, next_state = (0, 0, 0, 0), (1, 1, 1, 1)
    agent.q_table[next_state] = np.array([10.0, 10.0])  # would inflate target if used
    agent.update(state, action=0, reward=1.0, next_state=next_state, done=True)
    # With done=True, future_q must be 0, so Q should equal alpha * (reward - 0)
    expected = agent.alpha * 1.0
    assert abs(agent.q_table[state][0] - expected) < 1e-9
