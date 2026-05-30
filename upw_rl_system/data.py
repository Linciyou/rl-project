"""Data loading and preprocessing for UPW process history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import DataConfig


@dataclass(frozen=True)
class PreparedData:
    transition_inputs: torch.Tensor
    transition_targets: torch.Tensor
    scaled_states: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]
    raw_rows: int
    filtered_rows: int


def validate_required_columns(frame: pd.DataFrame, config: DataConfig) -> None:
    required = {config.arm_column, config.action_column, *config.features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"CSV is missing required columns: {missing_text}")


def load_and_preprocess_data(config: DataConfig) -> PreparedData:
    frame = pd.read_csv(config.csv_path)
    validate_required_columns(frame, config)

    filtered = frame[frame[config.arm_column] == config.arm].copy().reset_index(drop=True)
    if len(filtered) < 2:
        raise ValueError(
            f"Need at least 2 rows for arm '{config.arm}', found {len(filtered)}."
        )

    feature_names = list(config.features)
    filtered[feature_names] = filtered[feature_names].ffill().fillna(0.0)

    action_values = (
        filtered[config.action_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(config.positive_action_value.lower())
        .astype(np.float32)
        .to_numpy()
        .reshape(-1, 1)
    )

    scaler = StandardScaler()
    scaled_states = scaler.fit_transform(filtered[feature_names]).astype(np.float32)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for idx in range(len(scaled_states) - 1):
        inputs.append(np.concatenate([scaled_states[idx], action_values[idx]]))
        targets.append(scaled_states[idx + 1])

    transition_inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float32).unsqueeze(1)
    transition_targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32)

    return PreparedData(
        transition_inputs=transition_inputs,
        transition_targets=transition_targets,
        scaled_states=scaled_states,
        scaler=scaler,
        feature_names=feature_names,
        raw_rows=len(frame),
        filtered_rows=len(filtered),
    )
