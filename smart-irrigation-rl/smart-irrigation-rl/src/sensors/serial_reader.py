"""
serial_reader.py
-----------------
Real-time data ingestion layer that reads sensor readings streamed by the
Arduino Uno firmware over a USB-serial connection (PySerial), parses the
CSV/JSON payload, and converts it into the same `SensorReading` structure
used by the simulator -- so the RL agent and analytics layer are agnostic
to whether data originates from hardware or simulation.

Expected Arduino serial payload (one line per reading, CSV):
    soil_moisture,temperature,humidity,light_intensity,rain,battery,pressure
Example:
    54.30,27.10,61.50,512.00,0,4.01,1013.20
"""

from __future__ import annotations

import csv
import io
import time
from typing import Optional

from src.sensors.sensor_simulator import SensorReading
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import serial  # pyserial
    from serial import SerialException
except ImportError:  # pragma: no cover - allows the module to be imported
    serial = None    # without pyserial installed (e.g., in CI / no hardware)
    SerialException = Exception


class SerialReadError(Exception):
    """Raised when a serial payload cannot be parsed into a SensorReading."""


class ArduinoSerialReader:
    """
    Wraps a PySerial connection to an Arduino Uno running the sensor-fusion
    firmware, exposing a simple `read()` method that returns a validated
    `SensorReading`.

    Parameters
    ----------
    port : str
        Serial port, e.g. "COM3" (Windows) or "/dev/ttyUSB0" (Linux).
    baud_rate : int
        Must match the `Serial.begin()` rate set in the Arduino sketch.
    timeout : float
        Read timeout in seconds.
    """

    EXPECTED_FIELDS = 7

    def __init__(self, port: str = "COM3", baud_rate: int = 9600, timeout: float = 2.0):
        if serial is None:
            raise ImportError(
                "pyserial is not installed. Run `pip install pyserial` to use "
                "ArduinoSerialReader, or use SensorSimulator for offline mode."
            )
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._conn: Optional["serial.Serial"] = None
        self._step_counter = 0

    def connect(self) -> None:
        """Open the serial connection. Raises SerialException on failure."""
        try:
            self._conn = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            time.sleep(2.0)  # allow Arduino to reset after connection opens
            logger.info("Connected to Arduino on %s @ %d baud", self.port, self.baud_rate)
        except SerialException as exc:
            logger.error("Failed to open serial port %s: %s", self.port, exc)
            raise

    def disconnect(self) -> None:
        """Close the serial connection if open."""
        if self._conn and self._conn.is_open:
            self._conn.close()
            logger.info("Serial connection to %s closed.", self.port)

    def _parse_line(self, line: str) -> SensorReading:
        reader = csv.reader(io.StringIO(line.strip()))
        try:
            row = next(reader)
        except StopIteration as exc:
            raise SerialReadError("Empty serial line received.") from exc

        if len(row) != self.EXPECTED_FIELDS:
            raise SerialReadError(
                f"Expected {self.EXPECTED_FIELDS} fields, got {len(row)}: {row}"
            )

        try:
            soil_moisture, temperature, humidity, light_intensity, rain, battery, pressure = (
                float(v) for v in row
            )
        except ValueError as exc:
            raise SerialReadError(f"Non-numeric value in payload: {row}") from exc

        reading = SensorReading(
            timestamp_step=self._step_counter,
            soil_moisture=round(soil_moisture, 2),
            temperature=round(temperature, 2),
            humidity=round(humidity, 2),
            light_intensity=round(light_intensity, 2),
            rain_detected=bool(int(rain)),
            battery_voltage=round(battery, 3),
            ambient_pressure=round(pressure, 2),
        )
        self._step_counter += 1
        return reading

    def read(self, retries: int = 3) -> SensorReading:
        """
        Read and parse a single line from the serial buffer.

        Parameters
        ----------
        retries : int
            Number of retry attempts on malformed or empty lines.

        Returns
        -------
        SensorReading

        Raises
        ------
        SerialReadError
            If no valid reading could be parsed after all retries.
        """
        if self._conn is None or not self._conn.is_open:
            raise SerialReadError("Serial connection is not open. Call connect() first.")

        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                raw = self._conn.readline().decode("utf-8", errors="ignore")
                if not raw.strip():
                    raise SerialReadError("No data received (timeout or empty line).")
                return self._parse_line(raw)
            except (SerialReadError, SerialException) as exc:
                last_error = exc
                logger.warning("Serial read attempt %d/%d failed: %s", attempt, retries, exc)
                time.sleep(0.2)

        raise SerialReadError(f"Failed to read valid sensor data after {retries} retries.") from last_error

    def __enter__(self) -> "ArduinoSerialReader":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
