"""
sensor_simulator.py
--------------------
Simulates the 7+ IoT sensor suite (DHT11 temperature/humidity, LDR light,
capacitive soil moisture, plus derived sensors) used by the physical Arduino
Uno rig. This lets the RL agent be trained and the analytics dashboard be
demoed end-to-end without physical hardware attached.

The simulator models a simple day/night cycle for light and temperature,
gradual soil-moisture depletion (evapotranspiration), and a moisture bump
whenever an irrigation action is applied -- mirroring real field dynamics
closely enough for policy learning and demonstration purposes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class SensorReading:
    """Unified sensor state vector fed into the RL agent."""

    timestamp_step: int
    soil_moisture: float      # % (capacitive soil moisture sensor)
    temperature: float        # deg C (DHT11)
    humidity: float           # % relative humidity (DHT11)
    light_intensity: float    # lux-proxy 0-1023 (LDR, analog)
    rain_detected: bool       # rain sensor (digital)
    battery_voltage: float    # V (solar/battery monitor)
    ambient_pressure: float   # hPa (barometric sensor, optional module)

    def to_dict(self) -> Dict:
        return asdict(self)


class SensorSimulator:
    """
    Generates realistic, temporally-coherent synthetic sensor readings.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    steps_per_day : int
        Number of simulation steps representing one full day/night cycle
        (default 96 == one reading every 15 minutes).
    """

    def __init__(self, seed: int = 42, steps_per_day: int = 96):
        self._rng = random.Random(seed)
        self.steps_per_day = steps_per_day
        self._soil_moisture = 55.0  # start at a comfortable mid-range value

    def _day_phase(self, step: int) -> float:
        """Returns a 0-1 sinusoidal phase representing time-of-day (peak at noon)."""
        angle = 2 * math.pi * (step % self.steps_per_day) / self.steps_per_day
        return max(0.0, math.sin(angle - math.pi / 2) * -1)  # peaks at midday

    def reset(self) -> None:
        """Reset internal state (e.g., at the start of a new training episode)."""
        self._soil_moisture = 55.0

    def read(self, step: int, irrigated_last_step: bool = False) -> SensorReading:
        """
        Produce the next sensor reading.

        Parameters
        ----------
        step : int
            Current simulation step index (used to derive day/night cycle).
        irrigated_last_step : bool
            Whether the RL agent irrigated on the previous step; used to
            model the resulting rise in soil moisture.

        Returns
        -------
        SensorReading
        """
        day_phase = self._day_phase(step)  # 0 (night) -> 1 (midday)

        # --- Temperature: 18C at night, up to 36C at midday, plus noise ---
        temperature = 18 + 18 * day_phase + self._rng.gauss(0, 1.0)

        # --- Humidity: inversely related to temperature, plus noise ---
        humidity = 80 - 40 * day_phase + self._rng.gauss(0, 3.0)
        humidity = min(100.0, max(10.0, humidity))

        # --- Light intensity (LDR analog 0-1023): bright at midday ---
        light_intensity = 1023 * day_phase + self._rng.gauss(0, 20)
        light_intensity = min(1023.0, max(0.0, light_intensity))

        # --- Rain event: rare random occurrence ---
        rain_detected = self._rng.random() < 0.03

        # --- Soil moisture dynamics ---
        evapotranspiration = 0.4 + 0.6 * day_phase  # higher loss when hot/sunny
        self._soil_moisture -= evapotranspiration
        if irrigated_last_step:
            self._soil_moisture += 18.0  # irrigation bump
        if rain_detected:
            self._soil_moisture += 12.0
        self._soil_moisture += self._rng.gauss(0, 0.5)
        self._soil_moisture = min(100.0, max(0.0, self._soil_moisture))

        # --- Battery voltage: slow drain, solar recharge at midday ---
        battery_voltage = 3.7 + 0.5 * day_phase + self._rng.gauss(0, 0.02)

        # --- Ambient pressure: mostly stable, small noise ---
        ambient_pressure = 1013 + self._rng.gauss(0, 2.0)

        return SensorReading(
            timestamp_step=step,
            soil_moisture=round(self._soil_moisture, 2),
            temperature=round(temperature, 2),
            humidity=round(humidity, 2),
            light_intensity=round(light_intensity, 2),
            rain_detected=rain_detected,
            battery_voltage=round(battery_voltage, 3),
            ambient_pressure=round(ambient_pressure, 2),
        )
