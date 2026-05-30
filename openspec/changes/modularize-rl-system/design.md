## Context

The project is being built locally on this Windows computer as a complete UPW reinforcement-learning maintenance scheduling system. The expected outcome is a maintainable Python project with a clear command-line entry point, modular system internals, explicit dependency metadata, and README documentation.

The system will support a workflow where CSV process history is loaded, transformed into model training data, used to train a PINN/LSTM digital twin, used to construct a Gymnasium simulation environment, and finally used to train a PPO policy that generates a weekly mixed-bed regeneration schedule.

## Goals / Non-Goals

**Goals:**

- Build `化工廠.py` as the primary user-facing launcher for the project.
- Build a Python package named `upw_rl_system` with explicit modules for data processing, digital-twin modeling, simulation, training orchestration, reporting, and CLI behavior.
- Provide `python 化工廠.py ...` and `python -m upw_rl_system ...` as supported command-line entry points.
- Make CSV path, plant arm, training durations, output directory, random seed, and core simulation settings configurable through CLI arguments.
- Save model and schedule artifacts into a configured output directory.
- Document setup, CSV schema, usage, outputs, architecture, and limitations in README.

**Non-Goals:**

- Do not build a web UI, database, or cloud deployment layer.
- Do not position the result as a plant safety control system.
- Do not add unrelated model families beyond the planned PINN/LSTM plus PPO workflow.

## Decisions

1. Use `化工廠.py` as the project launcher and `upw_rl_system` as the implementation package.

   Rationale: `化工廠.py` gives the project a simple recognizable entry file, while the package keeps the actual system maintainable and importable.

   Alternative considered: put all logic directly in `化工廠.py`. That would make the first run simple but would make the system harder to extend, document, and test.

2. Split modules by workflow responsibility.

   Planned modules:

   - `config.py`: dataclasses for feature names and training configuration.
   - `data.py`: CSV loading, validation, filtering, scaling, and transition tensor creation.
   - `model.py`: PINN/LSTM digital twin and training function.
   - `environment.py`: Gymnasium environment backed by the trained digital twin.
   - `pipeline.py`: orchestration for training, schedule rollout, and artifact writing.
   - `cli.py`: argument parsing and console entry point.
   - `__main__.py`: package execution bridge.

   Rationale: these boundaries map directly to the project workflow and keep future changes localized.

3. Use `argparse` for CLI behavior.

   Rationale: the standard library is enough for this command surface and avoids introducing an extra dependency.

   Alternative considered: Typer or Click. They are useful for larger CLIs but are unnecessary for the current scope.

4. Keep generated artifacts under a configurable output directory.

   Rationale: PPO models, digital-twin checkpoints, preprocessing metadata, and schedule files should be grouped consistently instead of being scattered in the repository root.

5. Keep dependency installation outside source code.

   Rationale: dependency installation belongs in `requirements.txt` and README instructions. Runtime source files should contain system behavior, not environment setup commands.

## Risks / Trade-offs

- Non-ASCII launcher filename -> Keep the package entry point documented as an alternative for terminals or environments that have trouble with the Chinese filename.
- Model quality depends on CSV history quality -> Document required columns and keep configuration explicit.
- RL training can be slow -> Expose training timesteps and epochs as CLI options so the user can choose quick development runs or longer training runs.
- Generated artifacts may grow large -> Ignore output directories in git and keep generated files under `outputs/`.

## Development Plan

1. Create OpenSpec proposal, design, spec, and tasks for the UPW RL system.
2. Create the `upw_rl_system` package and `化工廠.py` launcher.
3. Implement configuration, preprocessing, model, environment, pipeline, and CLI modules.
4. Add dependency metadata and generated-output ignore rules.
5. Write README documentation for local setup and usage.
6. Record OpenSpec task completion and final development status.

## Open Questions

- The real CSV file is not stored in the project. The CLI requires the user to provide the input path with `--csv`.
