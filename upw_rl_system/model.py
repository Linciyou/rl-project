"""PINN/LSTM digital twin for UPW process dynamics."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from .config import DigitalTwinTrainingConfig


class UPWPinnLSTM(nn.Module):
    """Predict the next scaled process state from state plus action."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        output, hidden = self.lstm(inputs, hidden)
        delta_state = self.fc(output[:, -1, :])
        current_state = inputs[:, -1, :-1]
        next_state = current_state + delta_state
        return next_state, delta_state, hidden


def train_digital_twin(
    transition_inputs: torch.Tensor,
    transition_targets: torch.Tensor,
    state_dim: int,
    toc_idx: int,
    config: DigitalTwinTrainingConfig,
    logger: Callable[[str], None] | None = print,
) -> UPWPinnLSTM:
    model = UPWPinnLSTM(
        input_dim=state_dim + 1,
        output_dim=state_dim,
        hidden_dim=config.hidden_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        next_state_pred, delta_state_pred, _ = model(transition_inputs)
        actions = transition_inputs[:, 0, -1]
        delta_toc = delta_state_pred[:, toc_idx]

        loss_data = mse(next_state_pred, transition_targets)
        penalty_no_maint_min = torch.clamp(-delta_toc, min=0.0) * (1 - actions)
        penalty_no_maint_max = (
            torch.clamp(delta_toc - config.max_no_maintenance_toc_rise, min=0.0)
            * (1 - actions)
        )
        penalty_maint = torch.clamp(delta_toc + config.maintenance_toc_drop, min=0.0) * actions
        loss_physics = config.physics_weight * torch.mean(
            penalty_no_maint_min + penalty_no_maint_max + penalty_maint
        )

        total_loss = loss_data + loss_physics
        total_loss.backward()
        optimizer.step()

        if logger and config.log_every > 0 and (epoch + 1) % config.log_every == 0:
            logger(
                "PINN epoch "
                f"{epoch + 1:3d}/{config.epochs} | "
                f"loss={total_loss.item():.4f} | "
                f"physics={loss_physics.item():.4f}"
            )

    return model
