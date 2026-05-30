"""End-to-end orchestration for the UPW RL system."""

from __future__ import annotations

import csv
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .config import PipelineConfig
from .data import PreparedData, load_and_preprocess_data
from .environment import PINNSimulatedUPWEnv
from .model import UPWPinnLSTM, train_digital_twin


@dataclass(frozen=True)
class ScheduleRow:
    week: int
    action: int
    real_toc: float
    reward: float
    no_maint_weeks: int


@dataclass(frozen=True)
class PipelineResult:
    schedule: list[ScheduleRow]
    regeneration_weeks: list[int]
    output_dir: Path
    raw_rows: int
    filtered_rows: int


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_environment(
    model: UPWPinnLSTM,
    data: PreparedData,
    config: PipelineConfig,
) -> PINNSimulatedUPWEnv:
    return PINNSimulatedUPWEnv(
        pinn_model=model,
        initial_states=data.scaled_states,
        scaler=data.scaler,
        feature_names=data.feature_names,
        config=config.simulation,
    )


def save_digital_twin(
    model: UPWPinnLSTM,
    data: PreparedData,
    config: PipelineConfig,
) -> None:
    output_dir = config.output.output_dir
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": data.feature_names,
            "state_dim": len(data.feature_names),
            "hidden_dim": config.digital_twin.hidden_dim,
            "data_config": asdict(config.data),
            "digital_twin_config": asdict(config.digital_twin),
            "simulation_config": asdict(config.simulation),
        },
        output_dir / "digital_twin.pt",
    )
    with (output_dir / "preprocessing.pkl").open("wb") as handle:
        pickle.dump(
            {
                "scaler": data.scaler,
                "feature_names": data.feature_names,
                "data_config": asdict(config.data),
            },
            handle,
        )


def save_schedule(schedule: list[ScheduleRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["week", "action", "real_toc", "reward", "no_maint_weeks"],
        )
        writer.writeheader()
        for row in schedule:
            writer.writerow(asdict(row))


def rollout_policy(
    policy: PPO,
    env: PINNSimulatedUPWEnv,
    seed: int,
) -> list[ScheduleRow]:
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    rows: list[ScheduleRow] = []

    while not (done or truncated):
        action, _ = policy.predict(obs, deterministic=True)
        action_int = int(np.asarray(action).item())
        obs, reward, done, truncated, info = env.step(action_int)
        rows.append(
            ScheduleRow(
                week=env.step_count,
                action=action_int,
                real_toc=float(info["real_toc"]),
                reward=float(reward),
                no_maint_weeks=int(info["no_maint_weeks"]),
            )
        )

    return rows


def print_schedule(schedule: list[ScheduleRow], printer=print) -> None:
    printer("")
    printer("Weekly simulated schedule")
    printer("-------------------------")
    for row in schedule:
        decision = "regenerate MB" if row.action == 1 else "continue"
        printer(
            f"week={row.week:02d} action={row.action} "
            f"toc={row.real_toc:.2f} ppb reward={row.reward:.2f} "
            f"no_maint_weeks={row.no_maint_weeks} decision={decision}"
        )

    regeneration_weeks = [row.week for row in schedule if row.action == 1]
    printer("-------------------------")
    printer(f"regeneration_count={len(regeneration_weeks)}")
    printer(f"regeneration_weeks={regeneration_weeks}")


def run_pipeline(config: PipelineConfig, printer=print) -> PipelineResult:
    set_random_seed(config.seed)
    output_dir = config.output.output_dir
    if config.output.save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

    printer("Step 1/4: loading and preprocessing CSV data")
    data = load_and_preprocess_data(config.data)
    state_dim = len(data.feature_names)
    toc_idx = data.feature_names.index("TOC_ppb")

    printer("Step 2/4: training PINN/LSTM digital twin")
    digital_twin = train_digital_twin(
        transition_inputs=data.transition_inputs,
        transition_targets=data.transition_targets,
        state_dim=state_dim,
        toc_idx=toc_idx,
        config=config.digital_twin,
        logger=printer,
    )
    if config.output.save_artifacts:
        save_digital_twin(digital_twin, data, config)

    printer("Step 3/4: creating simulated UPW environment")
    env = build_environment(digital_twin, data, config)

    printer("Step 4/4: training PPO policy")
    policy = PPO(
        "MlpPolicy",
        env,
        verbose=config.ppo.verbose,
        learning_rate=config.ppo.learning_rate,
        seed=config.seed,
    )
    policy.learn(total_timesteps=config.ppo.total_timesteps)
    if config.output.save_artifacts:
        policy.save(str(output_dir / "ppo_policy.zip"))

    schedule = rollout_policy(policy, env, seed=config.seed)
    print_schedule(schedule, printer=printer)
    if config.output.save_artifacts:
        save_schedule(schedule, output_dir / "schedule.csv")

    regeneration_weeks = [row.week for row in schedule if row.action == 1]
    return PipelineResult(
        schedule=schedule,
        regeneration_weeks=regeneration_weeks,
        output_dir=output_dir,
        raw_rows=data.raw_rows,
        filtered_rows=data.filtered_rows,
    )
