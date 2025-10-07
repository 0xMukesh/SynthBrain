import torch
from typing import Literal

from src.model import Critic


def calculate_grad_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels,
    device: Literal["cuda", "cpu"],
) -> torch.Tensor:
    alpha = torch.rand((real.size(0), 1, 1, 1)).to(device)

    mixed = alpha * real + (1 - alpha) * fake
    mixed.requires_grad_(True)

    critic_mixed = critic(mixed, labels)

    gradient = torch.autograd.grad(
        inputs=mixed,
        outputs=critic_mixed,
        create_graph=True,
        grad_outputs=torch.ones_like(critic_mixed),
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)

    gradient_norm = torch.norm(gradient, p=2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
