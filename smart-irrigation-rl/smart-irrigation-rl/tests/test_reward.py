"""Unit tests for src.rl.reward."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl.reward import compute_sis, symmetric_reward, SISWeights
from src.sensors.sensor_simulator import SensorReading


def _reading(**overrides) -> SensorReading:
    base = dict(
        timestamp_step=0,
        soil_moisture=55.0,
        temperature=24.0,
        humidity=55.0,
        light_intensity=500.0,
        rain_detected=False,
        battery_voltage=4.0,
        ambient_pressure=1013.0,
    )
    base.update(overrides)
    return SensorReading(**base)


def test_sis_is_high_for_ideal_conditions():
    reading = _reading()
    score = compute_sis(reading)
    assert score > 90.0


def test_sis_is_low_for_poor_conditions():
    reading = _reading(soil_moisture=5.0, temperature=45.0, humidity=5.0, light_intensity=1023.0)
    score = compute_sis(reading)
    assert score < 40.0


def test_sis_bounded_between_0_and_100():
    for moisture in (0, 25, 50, 75, 100):
        reading = _reading(soil_moisture=moisture)
        score = compute_sis(reading)
        assert 0.0 <= score <= 100.0


def test_reward_penalizes_symmetric_deviation_equally():
    dry = _reading(soil_moisture=45.0)   # 10 below ideal (55)
    wet = _reading(soil_moisture=65.0)   # 10 above ideal (55)
    r_dry = symmetric_reward(dry, action=0)
    r_wet = symmetric_reward(wet, action=0)
    # Moisture-deviation component should be identical in magnitude;
    # SIS band is also symmetric around the same target, so total rewards
    # should be very close (within a small tolerance).
    assert abs(r_dry - r_wet) < 0.05


def test_reward_penalizes_irrigation_during_rain():
    rainy = _reading(rain_detected=True)
    r_irrigate = symmetric_reward(rainy, action=1)
    r_hold = symmetric_reward(rainy, action=0)
    assert r_irrigate < r_hold


def test_reward_penalizes_water_cost():
    dry_reading = _reading(soil_moisture=55.0)
    r_action = symmetric_reward(dry_reading, action=1, water_cost_per_action=2.0)
    r_no_action = symmetric_reward(dry_reading, action=0, water_cost_per_action=2.0)
    assert r_action < r_no_action
