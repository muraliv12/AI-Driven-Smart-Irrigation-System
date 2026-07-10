"""
main.py
-------
Command-line entrypoint for the AI-Driven Smart Irrigation System.

Usage
-----
    python -m src.main train                 # train a new Q-Learning agent
    python -m src.main evaluate               # evaluate a saved agent vs. baseline
    python -m src.main dashboard               # regenerate plots from existing logs
    python -m src.main live --port COM3        # stream live decisions from Arduino
"""

from __future__ import annotations

import argparse
import os
import sys

from src.analytics.dashboard import (
    plot_reward_convergence,
    plot_sis_trend,
    plot_water_usage_comparison,
    plot_irrigation_timeline,
)
from src.rl.evaluate import compare_to_baseline, run_single_episode_timeline
from src.rl.q_learning_agent import QLearningAgent
from src.rl.reward import compute_sis
from src.rl.train import train_agent
from src.rl.environment import IrrigationEnv
from src.sensors.serial_reader import ArduinoSerialReader
from src.utils.config_loader import load_config, ConfigError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    agent, env, history = train_agent(config, checkpoint_every=args.checkpoint_every)

    plot_reward_convergence(history, os.path.join(config.paths.outputs_dir, "reward_convergence.png"))
    plot_sis_trend(history, os.path.join(config.paths.outputs_dir, "sis_trend.png"))
    logger.info("Training artifacts written to %s", config.paths.outputs_dir)


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    env = IrrigationEnv(config.rl, config.thresholds, seed=config.rl.random_seed)
    agent = QLearningAgent(n_actions=env.N_ACTIONS, config=config.rl)

    if not os.path.exists(config.paths.q_table_file):
        logger.error("No trained Q-table found at %s. Run `train` first.", config.paths.q_table_file)
        sys.exit(1)
    agent.load(config.paths.q_table_file)

    summary, rl_history, baseline_history = compare_to_baseline(config, agent, n_episodes=args.episodes)
    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k:28s}: {v}")

    plot_water_usage_comparison(
        rl_history, baseline_history, os.path.join(config.paths.outputs_dir, "water_usage_comparison.png")
    )

    eval_env = IrrigationEnv(config.rl, config.thresholds, seed=config.rl.random_seed + 1)
    steps, moisture, actions = run_single_episode_timeline(agent, eval_env)
    plot_irrigation_timeline(
        steps, moisture, actions, os.path.join(config.paths.outputs_dir, "irrigation_timeline.png")
    )


def cmd_live(args: argparse.Namespace) -> None:
    """Stream real Arduino sensor data and print the trained agent's decisions."""
    config = load_config(args.config)
    env = IrrigationEnv(config.rl, config.thresholds, seed=config.rl.random_seed)
    agent = QLearningAgent(n_actions=env.N_ACTIONS, config=config.rl)

    if not os.path.exists(config.paths.q_table_file):
        logger.error("No trained Q-table found. Run `train` first.")
        sys.exit(1)
    agent.load(config.paths.q_table_file)

    reader = ArduinoSerialReader(port=args.port, baud_rate=config.serial.baud_rate,
                                  timeout=config.serial.timeout)
    with reader:
        logger.info("Streaming live sensor data from %s. Press Ctrl+C to stop.", args.port)
        try:
            while True:
                reading = reader.read()
                state = env._discretize(reading)
                action = agent.select_action(state, greedy=True)
                sis = compute_sis(reading)
                decision = "IRRIGATE" if action == 1 else "HOLD"
                print(f"[step] moisture={reading.soil_moisture:5.1f}% "
                      f"temp={reading.temperature:5.1f}C SIS={sis:5.1f} -> {decision}")
        except KeyboardInterrupt:
            logger.info("Live streaming stopped by user.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-Driven Smart Irrigation System (Q-Learning)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train a new Q-Learning agent.")
    p_train.add_argument("--checkpoint-every", type=int, default=100)
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained agent vs. rule-based baseline.")
    p_eval.add_argument("--episodes", type=int, default=50)
    p_eval.set_defaults(func=cmd_evaluate)

    p_live = subparsers.add_parser("live", help="Run the trained agent on live Arduino sensor data.")
    p_live.add_argument("--port", default="COM3", help="Serial port for the Arduino Uno.")
    p_live.set_defaults(func=cmd_live)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
