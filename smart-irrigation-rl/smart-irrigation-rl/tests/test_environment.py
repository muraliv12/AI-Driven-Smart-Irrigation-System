"""Unit tests for src.rl.environment."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl.environment import IrrigationEnv
from src.utils.config_loader import RLConfig, SensorThresholds


def _make_env(seed=1):
    rl_cfg = RLConfig(max_steps_per_episode=20, n_moisture_bins=5, n_temp_bins=4,
                       n_humidity_bins=4, n_light_bins=3, random_seed=seed)
    thresholds = SensorThresholds()
    return IrrigationEnv(rl_cfg, thresholds, seed=seed)


def test_reset_returns_valid_state_tuple():
    env = _make_env()
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 4
    m, t, h, l = state
    assert 0 <= m < env.cfg.n_moisture_bins
    assert 0 <= t < env.cfg.n_temp_bins
    assert 0 <= h < env.cfg.n_humidity_bins
    assert 0 <= l < env.cfg.n_light_bins


def test_step_returns_expected_tuple_shape():
    env = _make_env()
    env.reset()
    next_state, reward, done, info = env.step(0)
    assert isinstance(next_state, tuple) and len(next_state) == 4
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "reading" in info and "sis" in info and "water_used" in info


def test_episode_terminates_after_max_steps():
    env = _make_env()
    env.reset()
    done = False
    steps_taken = 0
    while not done:
        _, _, done, _ = env.step(0)
        steps_taken += 1
        assert steps_taken <= env.cfg.max_steps_per_episode
    assert steps_taken == env.cfg.max_steps_per_episode


def test_invalid_action_raises_value_error():
    env = _make_env()
    env.reset()
    try:
        env.step(2)
        assert False, "Expected ValueError for invalid action"
    except ValueError:
        pass


def test_water_used_flag_matches_action():
    env = _make_env()
    env.reset()
    _, _, _, info = env.step(1)
    assert info["water_used"] == 1
    env.reset()
    _, _, _, info = env.step(0)
    assert info["water_used"] == 0
