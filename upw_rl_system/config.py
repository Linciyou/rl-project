"""Configuration objects for the UPW RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_FEATURES: tuple[str, ...] = (
    "feed_SDI",
    "feed_conductivity_uScm",
    "resistivity_MOhm_cm",
    "TOC_ppb",
    "cation_leak_ppb",
    "anion_leak_ppb",
    "deltaP_bar",
)


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    arm: str = "Legacy_2passRO_MB"
    features: tuple[str, ...] = DEFAULT_FEATURES
    arm_column: str = "arm"
    action_column: str = "MB_regen"
    positive_action_value: str = "yes"


@dataclass(frozen=True)
class DigitalTwinTrainingConfig:
    epochs: int = 250
    hidden_dim: int = 64
    learning_rate: float = 0.003
    physics_weight: float = 10.0
    max_no_maintenance_toc_rise: float = 0.4
    maintenance_toc_drop: float = 1.5
    log_every: int = 50


@dataclass(frozen=True)
class SimulationConfig:
    max_steps: int = 52
    toc_baseline: float = 2.0
    toc_floor: float = 1.0
    failure_toc: float = 5.0
    warning_toc: float = 4.0
    healthy_reward: float = 1.0
    warning_penalty: float = 10.0
    failure_penalty: float = 100.0
    regeneration_cost: float = 15.0
    observation_low: float = -10.0
    observation_high: float = 10.0


@dataclass(frozen=True)
class PPOTrainingConfig:
    total_timesteps: int = 40_000
    learning_rate: float = 0.0005
    verbose: int = 0


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = Path("outputs")
    save_artifacts: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    digital_twin: DigitalTwinTrainingConfig = DigitalTwinTrainingConfig()
    simulation: SimulationConfig = SimulationConfig()
    ppo: PPOTrainingConfig = PPOTrainingConfig()
    output: OutputConfig = OutputConfig()
    seed: int = 42
