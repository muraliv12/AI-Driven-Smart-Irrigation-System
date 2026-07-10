# Arduino Uno Firmware Reference

This document describes the firmware contract that `src/sensors/serial_reader.py`
(`ArduinoSerialReader`) expects from the Arduino Uno. It is a **reference
sketch** describing the wiring and serial protocol used during development
of the physical rig — flash it to your own Uno to pair the trained agent
with real hardware.

## Wiring

| Sensor | Pin | Notes |
|---|---|---|
| DHT11 (temp/humidity) | D2 | Digital, requires `DHT` library |
| LDR (light) | A0 | Analog, voltage-divider with 10k resistor |
| Capacitive soil moisture | A1 | Analog |
| Rain sensor | D3 | Digital |
| Battery voltage divider | A2 | Analog, scaled 0–5V |
| BMP180/BMP280 (pressure) | I2C (A4/A5) | Optional module |
| Relay -> solenoid valve | D7 | Digital output, irrigation actuator |

## Serial Protocol

- Baud rate: `9600` (must match `config.yaml: serial.baud_rate`)
- One CSV line per reading, emitted every `read_interval_sec` (default 2s):
  ```
  soil_moisture,temperature,humidity,light_intensity,rain,battery,pressure
  ```
  Example:
  ```
  54.30,27.10,61.50,512.00,0,4.01,1013.20
  ```
- `rain` is `0` or `1` (digital pin state).
- The Python side (`ArduinoSerialReader`) reads exactly one line per
  `read()` call and blocks up to `timeout` seconds waiting for it.
- Optionally, the Arduino sketch can listen for a single ASCII byte
  (`'1'` = irrigate, `'0'` = hold) written back over serial by
  `src.main live` to actuate the relay — this loop is illustrated in
  the sequence below.

## Illustrative Sketch Outline

The reference Arduino sketch (`.ino`, flashed separately via the Arduino
IDE — not part of this Python package) performs, once per loop iteration:

1. Read DHT11 temperature + humidity.
2. Read LDR and soil-moisture analog pins, apply calibration scaling to
   0–100% / 0–1023 ranges.
3. Read the digital rain-sensor pin.
4. Read the battery voltage divider and scale to volts.
5. Read barometric pressure over I2C.
6. Print the seven values as a single CSV line terminated by `\n`.
7. Listen (non-blocking) for an incoming `'0'`/`'1'` byte and drive the
   relay pin accordingly.
8. Delay for the configured read interval before repeating.

Because the exact calibration constants (soil-moisture sensor wet/dry
reference voltages, LDR resistor value, etc.) are hardware-specific, they
are tuned per physical rig rather than hardcoded here — see the inline
comments left in your own `.ino` file's calibration section.
