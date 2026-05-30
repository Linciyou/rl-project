"""Gymnasium environment backed by the trained UPW digital twin."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

from .config import SimulationConfig
from .model import UPWPinnLSTM


class PINNSimulatedUPWEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        pinn_model: UPWPinnLSTM,
        initial_states: np.ndarray,
        scaler: StandardScaler,
        feature_names: list[str],
        config: SimulationConfig,
    ) -> None:
        super().__init__()
        self.pinn_model = pinn_model
        self.pinn_model.eval()
        self.initial_states = initial_states
        self.scaler = scaler
        self.feature_names = feature_names
        self.config = config

        self.state_dim = len(feature_names)
        self.toc_idx = feature_names.index("TOC_ppb")
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=config.observation_low,
            high=config.observation_high,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.consecutive_no_maintenance = 0
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.consecutive_no_maintenance = 0

        idx = int(self.np_random.integers(0, len(self.initial_states)))
        self.current_state = self.initial_states[idx].astype(np.float32)

        real_state = self.scaler.inverse_transform([self.current_state])[0]
        real_state[self.toc_idx] = self.config.toc_baseline
        self.current_state = self.scaler.transform([real_state])[0].astype(np.float32)

        hidden_size = self.pinn_model.lstm.hidden_size
        self.hidden_state = (
            torch.zeros(1, 1, hidden_size),
            torch.zeros(1, 1, hidden_size),
        )
        return self.current_state, {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1

        action_value = int(action)
        action_arr = np.array([action_value], dtype=np.float32)
        pinn_input = np.concatenate([self.current_state, action_arr])
        pinn_input_tensor = torch.as_tensor(pinn_input, dtype=torch.float32).view(1, 1, -1)

        with torch.no_grad():
            next_state_tensor, _, self.hidden_state = self.pinn_model(
                pinn_input_tensor,
                self.hidden_state,
            )

        self.current_state = next_state_tensor.numpy().flatten()
        real_state = self.scaler.inverse_transform([self.current_state])[0]
        real_toc = max(float(real_state[self.toc_idx]), self.config.toc_floor)
        real_state[self.toc_idx] = real_toc
        self.current_state = self.scaler.transform([real_state])[0].astype(np.float32)

        weeks_without_maint = self.consecutive_no_maintenance
        if action_value == 0:
            self.consecutive_no_maintenance += 1
        else:
            self.consecutive_no_maintenance = 0

        reward = self.config.healthy_reward
        terminated = False
        truncated = False

        if real_toc > self.config.failure_toc:
            reward -= self.config.failure_penalty
            terminated = True
        elif real_toc > self.config.warning_toc:
            reward -= self.config.warning_penalty

        if action_value == 1:
            reward -= self.config.regeneration_cost

        if self.step_count >= self.config.max_steps:
            truncated = True

        info = {
            "real_toc": real_toc,
            "no_maint_weeks": weeks_without_maint,
        }
        return self.current_state, float(reward), terminated, truncated, info
