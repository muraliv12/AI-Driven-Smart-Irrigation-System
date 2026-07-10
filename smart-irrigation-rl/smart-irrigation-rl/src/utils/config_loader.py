"""
config_loader.py
-----------------
Loads and validates the project YAML configuration file into a typed,
dot-accessible object so the rest of the codebase never has to touch
raw dictionaries or hardcode magic numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


@dataclass
class SerialConfig:
    port: str = "COM3"
    baud_rate: int = 9600
    timeout: float = 2.0
    read_interval_sec: float = 2.0


@dataclass
class SensorThresholds:
    soil_moisture_low: float = 30.0
    soil_moisture_high: float = 70.0
    temperature_low: float = 15.0
    temperature_high: float = 40.0
    humidity_low: float = 20.0
    humidity_high: float = 90.0
    light_low: float = 100.0
    light_high: float = 900.0


@dataclass
class RLConfig:
    n_episodes: int = 600
    max_steps_per_episode: int = 96          # e.g., 96 x 15-min steps = 1 day
    learning_rate: float = 0.15
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99
    n_moisture_bins: int = 5
    n_temp_bins: int = 4
    n_humidity_bins: int = 4
    n_light_bins: int = 3
    water_cost_per_action: float = 2.5
    random_seed: int = 42


@dataclass
class PathsConfig:
    data_dir: str = "data"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    q_table_file: str = "models/q_table.pkl"
    sensor_log_file: str = "data/sensor_log.csv"
    training_log_file: str = "outputs/training_log.csv"


@dataclass
class AppConfig:
    serial: SerialConfig = field(default_factory=SerialConfig)
    thresholds: SensorThresholds = field(default_factory=SensorThresholds)
    rl: RLConfig = field(default_factory=RLConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    log_level: str = "INFO"


def _build_dataclass(cls, data: Dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys safely."""
    if not data:
        return cls()
    valid_fields = {f for f in cls.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(path: str = "config/config.yaml") -> AppConfig:
    """
    Load the YAML configuration file into an AppConfig object.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    AppConfig
        Fully populated, typed configuration object.

    Raises
    ------
    ConfigError
        If the file does not exist or cannot be parsed.
    """
    if not os.path.exists(path):
        raise ConfigError(f"Configuration file not found at '{path}'.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {exc}") from exc

    return AppConfig(
        serial=_build_dataclass(SerialConfig, raw.get("serial", {})),
        thresholds=_build_dataclass(SensorThresholds, raw.get("thresholds", {})),
        rl=_build_dataclass(RLConfig, raw.get("rl", {})),
        paths=_build_dataclass(PathsConfig, raw.get("paths", {})),
        log_level=raw.get("log_level", "INFO"),
    )
