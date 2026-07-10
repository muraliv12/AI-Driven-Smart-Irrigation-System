"""
reward.py
---------
Defines the Smart Irrigation Score (SIS) -- a composite metric fusing soil
moisture, temperature, humidity, and light readings into a single scalar
in [0, 100] describing how favorable current conditions are -- and the
symmetric reward function used to train the Q-Learning agent.

The reward is "symmetric" in the sense that it penalizes both
under-watering (dry soil, crop-stress risk) and over-watering (wasted
water, root-rot risk) proportionally around an ideal soil-moisture band,
rather than only penalizing one direction.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.sensors.sensor_simulator import SensorReading


@dataclass
class SISWeights:
    """Relative importance of each factor in the Smart Irrigation Score."""

    moisture_weight: float = 0.55
    temperature_weight: float = 0.20
    humidity_weight: float = 0.15
    light_weight: float = 0.10


def _band_score(value: float, low: float, high: float) -> float:
    """
    Score a value in [0, 1]: 1.0 when inside [low, high], decaying linearly
    to 0.0 as the value moves further outside the band.
    """
    if low <= value <= high:
        return 1.0
    band_width = max(high - low, 1e-6)
    if value < low:
        distance = low - value
    else:
        distance = value - high
    return max(0.0, 1.0 - distance / band_width)


def compute_sis(reading: SensorReading, weights: SISWeights = SISWeights()) -> float:
    """
    Compute the Smart Irrigation Score (0-100) for a given sensor reading.

    Ideal bands used:
        soil moisture   : 45% - 65%
        temperature     : 18C - 30C
        humidity        : 40% - 70%
        light intensity : 200 - 800 (LDR analog units)

    Parameters
    ----------
    reading : SensorReading
    weights : SISWeights

    Returns
    -------
    float
        Smart Irrigation Score between 0 and 100 (higher == healthier
        growing conditions).
    """
    moisture_score = _band_score(reading.soil_moisture, 45.0, 65.0)
    temperature_score = _band_score(reading.temperature, 18.0, 30.0)
    humidity_score = _band_score(reading.humidity, 40.0, 70.0)
    light_score = _band_score(reading.light_intensity, 200.0, 800.0)

    sis = 100.0 * (
        weights.moisture_weight * moisture_score
        + weights.temperature_weight * temperature_score
        + weights.humidity_weight * humidity_score
        + weights.light_weight * light_score
    )
    return round(sis, 3)


def symmetric_reward(
    reading: SensorReading,
    action: int,
    water_cost_per_action: float = 1.0,
    ideal_moisture: float = 55.0,
    tolerance: float = 10.0,
) -> float:
    """
    Symmetric reward function used at every RL training step.

    Combines:
      1. A moisture-deviation penalty that is symmetric around the ideal
         soil-moisture level (equally penalizes too-dry and too-wet soil).
      2. The Smart Irrigation Score as a positive shaping term.
      3. A fixed water-cost penalty applied whenever the agent irrigates,
         encouraging water-efficient policies.

    Parameters
    ----------
    reading : SensorReading
        The sensor state *after* the action was applied.
    action : int
        0 == do not irrigate, 1 == irrigate.
    water_cost_per_action : float
        Penalty subtracted whenever action == 1.
    ideal_moisture : float
        Center of the target soil-moisture band.
    tolerance : float
        Half-width of the band used to normalize the deviation penalty.

    Returns
    -------
    float
        Scalar reward for this transition.
    """
    deviation = abs(reading.soil_moisture - ideal_moisture)
    # Symmetric quadratic penalty: identical cost for +N% and -N% deviation.
    moisture_penalty = -1.0 * min((deviation / tolerance) ** 2, 4.0)

    sis = compute_sis(reading)
    sis_term = sis / 100.0  # normalize to [0, 1]

    water_penalty = -water_cost_per_action if action == 1 else 0.0

    # Extra penalty if it's raining and the agent still irrigates (wasteful).
    rain_penalty = -1.5 if (reading.rain_detected and action == 1) else 0.0

    reward = moisture_penalty + sis_term + water_penalty + rain_penalty
    return round(reward, 4)
