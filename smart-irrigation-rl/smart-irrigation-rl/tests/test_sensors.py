"""Unit tests for src.sensors.sensor_simulator."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sensors.sensor_simulator import SensorSimulator, SensorReading


def test_reading_fields_within_bounds():
    sim = SensorSimulator(seed=1, steps_per_day=96)
    reading = sim.read(step=10)
    assert isinstance(reading, SensorReading)
    assert 0.0 <= reading.soil_moisture <= 100.0
    assert 0.0 <= reading.humidity <= 100.0
    assert 0.0 <= reading.light_intensity <= 1023.0
    assert isinstance(reading.rain_detected, bool)


def test_irrigation_increases_soil_moisture():
    sim = SensorSimulator(seed=2, steps_per_day=96)
    sim.read(step=0)
    before = sim._soil_moisture
    sim.read(step=1, irrigated_last_step=True)
    after = sim._soil_moisture
    assert after > before - 5  # irrigation bump should offset evapotranspiration


def test_reset_restores_initial_moisture():
    sim = SensorSimulator(seed=3, steps_per_day=96)
    for step in range(20):
        sim.read(step)
    sim.reset()
    assert sim._soil_moisture == 55.0


def test_reproducibility_with_same_seed():
    sim1 = SensorSimulator(seed=99, steps_per_day=96)
    sim2 = SensorSimulator(seed=99, steps_per_day=96)
    r1 = sim1.read(step=5)
    r2 = sim2.read(step=5)
    assert r1.to_dict() == r2.to_dict()
