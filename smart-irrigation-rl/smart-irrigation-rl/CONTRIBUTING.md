# Contributing to AI-Driven Smart Irrigation System

Thank you for considering a contribution to this project. This document
outlines the process for reporting issues, proposing changes, and
submitting pull requests.

## Getting Started

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/smart-irrigation-rl.git
   cd smart-irrigation-rl
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

- **Code style**: Follow PEP 8. Use type hints on all public functions and
  classes. Keep functions small and single-purpose.
- **Docstrings**: Every module, class, and public function must have a
  NumPy-style docstring describing parameters, return values, and behavior.
- **Tests**: Any new feature or bug fix must include corresponding tests in
  `tests/`. Run the full suite before opening a PR:
  ```bash
  pytest tests/ -v
  ```
- **Logging over print**: Use `src.utils.logger.get_logger(__name__)` for
  diagnostic output instead of bare `print()` statements (CLI-facing output
  in `src/main.py` is the one exception).
- **Configuration**: Do not hardcode tunable values (thresholds, RL
  hyperparameters, file paths) — add them to `config/config.yaml` and
  `src/utils/config_loader.py` instead.

## Commit Messages

Use clear, imperative-mood commit messages, e.g.:
```
Add epsilon-greedy annealing schedule to QLearningAgent
Fix soil-moisture discretization edge case at upper bound
```

## Pull Request Process

1. Ensure `pytest tests/ -v` passes locally with no failures.
2. Update `README.md` and/or `docs/` if your change affects usage,
   configuration, or architecture.
3. Add an entry to `CHANGELOG.md` under an "Unreleased" section.
4. Open a pull request against `main` with a description of the change,
   the motivation, and any relevant benchmark numbers (e.g., effect on
   water-usage reduction or SIS).
5. A maintainer will review your PR and may request changes before merging.

## Reporting Bugs

Please open an issue including:
- A clear description of the problem
- Steps to reproduce (config used, command run)
- Expected vs. actual behavior
- Python version and OS

## Code of Conduct

By participating in this project, you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
