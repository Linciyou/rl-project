## ADDED Requirements

### Requirement: Local project entry points
The system SHALL provide `化工廠.py` as the primary local launcher and SHALL also support package execution through `python -m upw_rl_system`.

#### Scenario: Run through project launcher
- **WHEN** a user runs `python 化工廠.py` with required CLI arguments
- **THEN** the system starts the UPW RL pipeline through the shared CLI behavior

#### Scenario: Run through package module
- **WHEN** a user runs `python -m upw_rl_system` with required CLI arguments
- **THEN** the system starts the same UPW RL pipeline

### Requirement: Explicit runtime dependencies
The system SHALL define Python dependencies outside of runtime source code.

#### Scenario: Dependency installation guidance
- **WHEN** a user reads the project setup files
- **THEN** the required packages are listed in `requirements.txt` and documented in README usage instructions

#### Scenario: Source files remain project code
- **WHEN** a user opens the Python source files
- **THEN** the files contain project behavior and do not embed package-installation commands as executable code

### Requirement: CSV preprocessing pipeline
The system SHALL load UPW process history from a CSV path supplied by the user, validate required columns, filter the selected plant arm, scale configured process features, map regeneration history into binary actions, and build one-step state transition training tensors.

#### Scenario: Valid CSV input
- **WHEN** the CSV contains the selected arm and all required feature, action, and arm columns
- **THEN** the preprocessing stage returns scaled states, transition inputs, transition targets, scaler metadata, and feature names

#### Scenario: Missing CSV columns
- **WHEN** the CSV is missing one or more required columns
- **THEN** the preprocessing stage fails with an error message that names the missing columns

### Requirement: Digital twin training
The system SHALL train a PINN/LSTM digital twin that predicts the next scaled process state from the current scaled state and binary regeneration action.

#### Scenario: Train digital twin with defaults
- **WHEN** the pipeline reaches the digital-twin training stage with preprocessed transitions
- **THEN** it trains the model using default epochs, learning rate, hidden size, and physics penalty settings unless CLI arguments override them

#### Scenario: Save digital twin artifact
- **WHEN** digital-twin training completes
- **THEN** the system writes a digital-twin checkpoint and preprocessing metadata into the configured output directory

### Requirement: UPW simulation environment
The system SHALL expose a Gymnasium-compatible environment that uses the trained digital twin to simulate weekly UPW state transitions under discrete regeneration actions.

#### Scenario: Environment reset
- **WHEN** the environment is reset
- **THEN** it returns a scaled observation with the configured TOC baseline applied and clears episode counters

#### Scenario: Environment step
- **WHEN** the agent chooses action `0` or `1`
- **THEN** the environment advances one simulated week, updates the no-maintenance counter, computes reward using TOC and regeneration cost rules, and returns Gymnasium step outputs

### Requirement: PPO policy training
The system SHALL train a Stable-Baselines3 PPO policy against the UPW simulation environment using configurable training parameters.

#### Scenario: Train PPO policy
- **WHEN** the pipeline reaches the RL training stage
- **THEN** it trains a PPO policy using default total timesteps and learning rate unless CLI arguments override them

#### Scenario: Save PPO policy artifact
- **WHEN** PPO training completes
- **THEN** the system writes the policy artifact into the configured output directory

### Requirement: Maintenance schedule report
The system SHALL roll out the trained policy deterministically and report the simulated weekly regeneration schedule.

#### Scenario: Console schedule output
- **WHEN** policy rollout completes
- **THEN** the system prints weekly TOC values, regeneration decisions, and a summary of regeneration weeks

#### Scenario: File schedule output
- **WHEN** an output directory is configured
- **THEN** the system writes the schedule as a CSV file containing week, action, real TOC, reward, and no-maintenance-week fields

### Requirement: README documentation
The system SHALL document the project purpose, architecture, setup, CSV schema, CLI usage, outputs, and limitations.

#### Scenario: User follows README
- **WHEN** a new user reads `README.md`
- **THEN** they can identify required dependencies, prepare input data, run the system, and understand generated artifacts without reading the source code first
