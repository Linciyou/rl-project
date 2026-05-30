"""Command-line interface for the UPW RL system."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DataConfig,
    DigitalTwinTrainingConfig,
    OutputConfig,
    PipelineConfig,
    PPOTrainingConfig,
    SimulationConfig,
)
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a UPW digital twin and PPO regeneration policy from CSV data.",
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to UPW history CSV.")
    parser.add_argument(
        "--arm",
        default="Legacy_2passRO_MB",
        help="Plant arm value to filter before training.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Directory for model and schedule artifacts.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument("--pinn-epochs", default=250, type=int, help="PINN training epochs.")
    parser.add_argument("--pinn-hidden-dim", default=64, type=int, help="PINN LSTM hidden size.")
    parser.add_argument("--pinn-lr", default=0.003, type=float, help="PINN learning rate.")
    parser.add_argument(
        "--pinn-log-every",
        default=50,
        type=int,
        help="Print PINN progress every N epochs. Use 0 to disable.",
    )

    parser.add_argument(
        "--ppo-timesteps",
        default=40_000,
        type=int,
        help="PPO total training timesteps.",
    )
    parser.add_argument("--ppo-lr", default=0.0005, type=float, help="PPO learning rate.")
    parser.add_argument("--ppo-verbose", default=0, type=int, help="Stable-Baselines3 verbosity.")

    parser.add_argument("--max-steps", default=52, type=int, help="Simulated weeks per rollout.")
    parser.add_argument(
        "--toc-baseline",
        default=2.0,
        type=float,
        help="TOC value applied at environment reset.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing model or schedule artifacts.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(csv_path=args.csv, arm=args.arm),
        digital_twin=DigitalTwinTrainingConfig(
            epochs=args.pinn_epochs,
            hidden_dim=args.pinn_hidden_dim,
            learning_rate=args.pinn_lr,
            log_every=args.pinn_log_every,
        ),
        simulation=SimulationConfig(
            max_steps=args.max_steps,
            toc_baseline=args.toc_baseline,
        ),
        ppo=PPOTrainingConfig(
            total_timesteps=args.ppo_timesteps,
            learning_rate=args.ppo_lr,
            verbose=args.ppo_verbose,
        ),
        output=OutputConfig(
            output_dir=args.output_dir,
            save_artifacts=not args.no_save,
        ),
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(config_from_args(args))
    return 0
