"""
Smart Irrigation RL
===================

An AI-driven smart irrigation controller that fuses multi-sensor IoT data
(soil moisture, temperature/humidity, light) into a unified state vector and
learns an optimal irrigation policy using tabular Q-Learning.

Modules
-------
sensors    : Real (PySerial/Arduino) and simulated sensor data acquisition.
rl         : Q-Learning agent, environment, and reward (Smart Irrigation Score).
analytics  : Real-time ingestion and Matplotlib dashboards.
utils      : Logging and configuration helpers.
"""

__version__ = "1.0.0"
__author__ = "Khuti"
