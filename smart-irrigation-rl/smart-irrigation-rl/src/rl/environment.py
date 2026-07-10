"""
environment.py
---------------
A Gym-like (but dependency-free) environment that wraps the sensor
simulator, fuses raw multi-sensor readings into a discretized state tuple
suitable for tabular Q-Learning, applies the irrigation action, and returns
the (next_state, reward, done, info) tuple each step.

State fusion: soil_moisture, temperature, humidity, light_intensity are
each bucketed into `n_*_bins` discrete levels, producing a compact state
space of size:
    n_moisture_bins * n_temp_bins * n_humidity_bins * n_light_bins
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.rl.reward import symmetric_reward, compute_sis
from src.sensors.sensor_simulator import SensorSimulator, SensorReading
from src.utils.config_loader import RLConfig, SensorThresholds


State = Tuple[int, int, int, int]


class IrrigationEnv:
    """
    Reinforcement-learning environment for the smart irrigation controller.

    Actions
    -------
    0 : Do not irrigate
    1 : Irrigate

    Parameters
    ----------
    rl_config : RLConfig
        Discretization bin counts, episode length, water cost, etc.
    thresholds : SensorThresholds
        Sensor operating ranges used to bound the discretization bins.
    seed : int
        Random seed forwarded to the underlying sensor simulator.
    """

    ACTION_NO_IRRIGATE = 0
    ACTION_IRRIGATE = 1
    N_ACTIONS = 2

    def __init__(self, rl_config: RLConfig, thresholds: SensorThresholds, seed: int = 42):
        self.cfg = rl_config
        self.thresholds = thresholds
        self.simulator = SensorSimulator(seed=seed, steps_per_day=rl_config.max_steps_per_episode)

        self._moisture_edges = np.linspace(0, 100, rl_config.n_moisture_bins + 1)
        self._temp_edges = np.linspace(
            thresholds.temperature_low - 5, thresholds.temperature_high + 5, rl_config.n_temp_bins + 1
        )
        self._humidity_edges = np.linspace(0, 100, rl_config.n_humidity_bins + 1)
        self._light_edges = np.linspace(0, 1023, rl_config.n_light_bins + 1)

        self.current_step = 0
        self._last_reading: SensorReading | None = None
        self._last_action = 0

    @property
    def state_space_size(self) -> Tuple[int, int, int, int]:
        return (
            self.cfg.n_moisture_bins,
            self.cfg.n_temp_bins,
            self.cfg.n_humidity_bins,
            self.cfg.n_light_bins,
        )

    def _discretize(self, reading: SensorReading) -> State:
        """Fuse continuous sensor values into a discrete state tuple."""
        m = int(np.clip(np.digitize(reading.soil_moisture, self._moisture_edges) - 1,
                         0, self.cfg.n_moisture_bins - 1))
        t = int(np.clip(np.digitize(reading.temperature, self._temp_edges) - 1,
                         0, self.cfg.n_temp_bins - 1))
        h = int(np.clip(np.digitize(reading.humidity, self._humidity_edges) - 1,
                         0, self.cfg.n_humidity_bins - 1))
        l = int(np.clip(np.digitize(reading.light_intensity, self._light_edges) - 1,
                         0, self.cfg.n_light_bins - 1))
        return (m, t, h, l)

    def reset(self) -> State:
        """Reset the environment at the start of a new training episode."""
        self.simulator.reset()
        self.current_step = 0
        self._last_action = 0
        self._last_reading = self.simulator.read(self.current_step, irrigated_last_step=False)
        return self._discretize(self._last_reading)

    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """
        Apply an action and advance the environment by one time step.

        Parameters
        ----------
        action : int
            0 (no irrigation) or 1 (irrigate).

        Returns
        -------
        next_state : State
        reward : float
        done : bool
        info : dict
            Contains the raw SensorReading, SIS score, and step index for
            logging / analytics purposes.
        """
        if action not in (self.ACTION_NO_IRRIGATE, self.ACTION_IRRIGATE):
            raise ValueError(f"Invalid action {action}; expected 0 or 1.")

        self.current_step += 1
        reading = self.simulator.read(self.current_step, irrigated_last_step=(action == 1))
        reward = symmetric_reward(
            reading, action, water_cost_per_action=self.cfg.water_cost_per_action
        )
        next_state = self._discretize(reading)

        done = self.current_step >= self.cfg.max_steps_per_episode
        info = {
            "reading": reading.to_dict(),
            "sis": compute_sis(reading),
            "step": self.current_step,
            "water_used": 1 if action == 1 else 0,
        }

        self._last_reading = reading
        self._last_action = action
        return next_state, reward, done, info
