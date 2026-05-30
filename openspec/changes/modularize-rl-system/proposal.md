## Why

This project needs a complete local UPW reinforcement-learning maintenance system that can be developed, explained, and maintained on this Windows computer. The system should start from a clear specification, provide a runnable project entry point, and include documentation for installing dependencies, preparing data, running the pipeline, and understanding outputs.

The goal is to build the project from zero into a structured Python system instead of leaving the workflow as an informal experiment.

## What Changes

- Create `化工廠.py` as the user-facing main launcher for the UPW RL maintenance system.
- Create a reusable `upw_rl_system/` Python package for the core workflow.
- Add a CLI that runs the complete pipeline from CSV input to trained PPO policy and simulated maintenance schedule.
- Separate data preprocessing, PINN/LSTM digital-twin training, Gymnasium environment behavior, PPO agent training, and schedule reporting.
- Add project documentation that explains installation, required CSV columns, command usage, outputs, and system structure.
- Add dependency and ignore-file metadata for repeatable local development on this computer.

## Capabilities

### New Capabilities

- `upw-rl-system`: Provides a reusable local system for training a UPW digital twin, training an RL maintenance policy, and generating a simulated regeneration schedule from CSV process data.

### Modified Capabilities

None.

## Impact

- Affected code: new `化工廠.py` launcher, new `upw_rl_system/` package modules, `requirements.txt`, `.gitignore`, and `README.md`.
- APIs: introduces command-line usage through `python 化工廠.py ...` and `python -m upw_rl_system ...`.
- Dependencies: defines the scientific Python stack required by the system.
- Documentation: creates README and OpenSpec records describing the project from initial request through completed development.
