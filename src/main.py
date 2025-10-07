import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List

from src.model import Generator, Critic
from src.dataset import BRISCDataset
from src.utils import calculate_grad_penalty
from src.constants import BRICS_CLASSES


@dataclass
class Config:
    epochs = 1
    batch_size = 64
    critic_iter = 3
    lambda_gp = 8

    lr_gen = 2e-4
    lr_critic = 1e-4

    latent_dim = 100
    base_features = 128
    num_blocks = 4
    img_channels = 1

    img_size = (128, 128)
    num_classes = len(BRICS_CLASSES)

    dataset_root = "./data"


config = Config()

transform = T.Compose(
    [T.Resize(config.img_size), T.Resize(config.img_size), T.Grayscale(), T.ToTensor()]
)

train_dataset = BRISCDataset(
    root=config.dataset_root, split="train", transform=transform
)
test_dataset = BRISCDataset(root=config.dataset_root, split="test", transform=transform)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=config.batch_size, shuffle=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator(
    img_channels=config.img_channels,
    latent_dim=config.latent_dim,
    base_features=config.base_features,
    num_blocks=config.num_blocks,
    num_classes=config.num_classes,
).to(device)
critic = Critic(
    img_channels=config.img_channels,
    base_features=config.base_features,
    num_blocks=config.num_blocks,
    num_classes=config.num_classes,
    img_size=config.img_size,
).to(device)

gen_optim = torch.optim.Adam(generator.parameters(), lr=config.lr_gen, betas=(0.0, 0.9))
critic_optim = torch.optim.Adam(
    critic.parameters(), lr=config.lr_critic, betas=(0.0, 0.9)
)

gen_losses: List[float] = []
critic_losses: List[float] = []

for epoch in range(config.epochs):
    for idx, (real, label) in enumerate(train_loader):
        label = label.to(device)
        real = real.to(device)

        batch_size = real.size(0)
        critic_loss_batch = 0.0

        for _ in range(config.critic_iter):
            noise = torch.randn((batch_size, config.latent_dim, 1, 1)).to(device)
            fake = generator(noise, label).detach()

            C_real = critic(real, label)
            C_fake = critic(fake, label)
            gp = calculate_grad_penalty(critic, real, fake, label, device)

            C_loss = torch.mean(C_fake) - torch.mean(C_real) + config.lambda_gp * gp
            critic_loss_batch += C_loss.item()

            critic_optim.zero_grad()
            C_loss.backward()
            critic_optim.step()

        noise = torch.randn((batch_size, config.latent_dim, 1, 1)).to(device)
        fake = generator(noise, label)

        G_loss = -torch.mean(critic(fake, label))

        gen_optim.zero_grad()
        G_loss.backward()
        gen_optim.step()

        gen_losses.append(G_loss.item())
        critic_losses.append(critic_loss_batch / config.critic_iter)

        if idx % 100 == 0:
            print(
                f"[epoch {epoch+1}/{config.epochs}, batch {idx}] "
                f"gen loss: {G_loss.item():.4f}, critic loss: {critic_loss_batch/config.critic_iter:.4f}"
            )

    print(f"epoch {epoch+1} completed")
