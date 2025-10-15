import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from PIL import Image
from typing import Literal, Dict, List, TypeAlias

from src.model import Critic, Generator
from src.dataset import BRISCDataset
from src.constants import BRICS_CLASSES

DeviceType: TypeAlias = Literal["cuda", "cpu"]


def calculate_grad_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    device: DeviceType,
) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand((batch_size, 1, 1, 1)).to(device)

    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    critic_interpolated = critic(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def visualize_random_real_samples_by_class(
    dataset: BRISCDataset, num_samples_per_class: int = 4
):
    num_classes = len(BRICS_CLASSES)
    samples_by_class: Dict[int, List[Image.Image]] = {i: [] for i in range(num_classes)}

    for img, label in dataset:
        if len(samples_by_class[label]) < num_samples_per_class:
            samples_by_class[label].append(TF.to_grayscale(TF.to_pil_image(img)))

        if all(
            len(samples) >= num_samples_per_class
            for samples in samples_by_class.values()
        ):
            break

    _, axes = plt.subplots(
        num_classes,
        num_samples_per_class,
        figsize=(num_samples_per_class * 2, num_classes * 2),
    )

    for i in range(num_classes):
        for j in range(num_samples_per_class):
            img = samples_by_class[i][j]
            ax: Axes = axes[i, j]
            np_arr = np.asarray(img)

            ax.imshow(np_arr, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if j == 0:
                ax.set_ylabel(
                    BRICS_CLASSES[i],
                    rotation=0,
                    labelpad=40,
                    fontsize=12,
                )

    plt.tight_layout()
    plt.show()


def visualize_output_samples(
    generator: Generator,
    latent_dim: int,
    device: DeviceType,
    num_samples_per_class: int = 4,
):
    generator.eval()

    num_classes = len(BRICS_CLASSES)
    outputs: List[List[torch.Tensor]] = []

    with torch.no_grad():
        for i in range(num_classes):
            row: List[torch.Tensor] = []

            for _ in range(num_samples_per_class):
                labels = torch.LongTensor([i]).to(device)
                noise = torch.randn((1, latent_dim, 1, 1)).to(device)
                output = generator(noise, labels)

                row.append(output)

            outputs.append(row)

    fig, axes = plt.subplots(
        num_classes,
        num_samples_per_class,
        figsize=(num_samples_per_class * 2, num_classes * 2),
    )

    if num_classes == 1:
        axes = axes.reshape(1, -1)
    elif num_samples_per_class == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_classes):
        for j in range(num_samples_per_class):
            ax: Axes = axes[i, j]

            sample = outputs[i][j]

            np_arr = sample.cpu().squeeze().numpy()
            np_arr = (np_arr + 1) / 2
            np_arr = np.clip(np_arr, 0, 1)

            ax.imshow(np_arr, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if j == 0:
                ax.set_ylabel(
                    BRICS_CLASSES[i], rotation=0, labelpad=40, fontsize=12, va="center"
                )

    plt.tight_layout()
    plt.show()
