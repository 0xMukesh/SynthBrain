import torch
import os
from torchvision import transforms as T
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List

from models.wdcgan import Critic, Generator
from dataset import BRISCDataset
from utils import calculate_grad_penalty
from constants import BRICS_CLASSES


@dataclass
class Config:
    epochs = 100
    batch_size = 64
    critic_iter = 5
    lambda_gp = 10

    lr_gen = 5e-5
    lr_critic = 1e-4
    beta1 = 0.0
    beta2 = 0.9

    latent_dim = 100
    base_features = 128
    num_blocks = 4
    img_channels = 1

    img_size = (128, 128)
    num_classes = len(BRICS_CLASSES)

    print_after_every = 10
    save_after_every = 5

    dataset_root = "/kaggle/input/brisc2025/brisc2025/classification_task"
    checkpoints_dir = "checkpoints"


config = Config()

os.makedirs(config.checkpoints_dir, exist_ok=True)

transform = T.Compose(
    [
        T.Resize(config.img_size),
        T.Grayscale(),
        T.RandomHorizontalFlip(p=0.3),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

train_dataset = BRISCDataset(
    root=config.dataset_root, split="train", transform=transform
)
test_dataset = BRISCDataset(root=config.dataset_root, split="test", transform=transform)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=config.batch_size, drop_last=True
)

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


gen_optim = torch.optim.Adam(
    generator.parameters(), lr=config.lr_gen, betas=(config.beta1, config.beta2)
)
critic_optim = torch.optim.Adam(
    critic.parameters(), lr=config.lr_critic, betas=(config.beta1, config.beta2)
)

gen_losses: List[float] = []
critic_losses: List[float] = []
epoch_gen_losses: List[float] = []
epoch_critic_losses: List[float] = []

best_gen_loss = float("inf")

for epoch in range(config.epochs):
    generator.train()
    critic.train()

    epoch_gen_loss = 0.0
    epoch_critic_loss = 0.0

    for idx, (real, label) in enumerate(train_loader):
        label = label.to(device)
        real = real.to(device)
        batch_size = real.size(0)

        critic_loss_accum = 0.0

        for _ in range(config.critic_iter):
            noise = torch.randn((batch_size, config.latent_dim, 1, 1)).to(device)

            with torch.no_grad():
                fake = generator(noise, label)

            C_real = critic(real, label)
            C_fake = critic(fake, label)
            gp = calculate_grad_penalty(critic, real, fake, label, device)

            C_loss = torch.mean(C_fake) - torch.mean(C_real) + config.lambda_gp * gp

            critic_optim.zero_grad()
            C_loss.backward()
            critic_optim.step()
            critic_loss_accum += C_loss.item()

        avg_critic_loss = critic_loss_accum / config.critic_iter

        noise = torch.randn((batch_size, config.latent_dim, 1, 1)).to(device)
        fake = generator(noise, label)

        G_loss = -torch.mean(critic(fake, label))

        gen_optim.zero_grad()
        G_loss.backward()
        gen_optim.step()

        gen_losses.append(G_loss.item())
        critic_losses.append(avg_critic_loss)
        epoch_gen_loss += G_loss.item()
        epoch_critic_loss += avg_critic_loss

        if idx % config.print_after_every == 0:
            print(
                f"[epoch {epoch+1}/{config.epochs}, batch {idx}/{len(train_loader)}] "
                f"gen loss: {G_loss.item():.4f}, critic loss: {avg_critic_loss:.4f}, "
            )

    avg_epoch_gen_loss = epoch_gen_loss / len(train_loader)
    avg_epoch_critic_loss = epoch_critic_loss / len(train_loader)
    epoch_gen_losses.append(avg_epoch_gen_loss)
    epoch_critic_losses.append(avg_epoch_critic_loss)

    print(f"\n{'='*60}")
    print(f"epoch {epoch+1} completed")
    print(f"avg gen loss: {avg_epoch_gen_loss:.4f}")
    print(f"avg critic loss: {avg_epoch_critic_loss:.4f}")
    print(f"{'='*60}\n")

    if avg_epoch_gen_loss < best_gen_loss:
        best_gen_loss = avg_epoch_gen_loss

        print(f"new best model! gen loss = {best_gen_loss:.4f}. saving checkpoint...")

        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "gen_optim_state_dict": gen_optim.state_dict(),
                "critic_optim_state_dict": critic_optim.state_dict(),
                "gen_loss": avg_epoch_gen_loss,
                "critic_loss": avg_epoch_critic_loss,
                "gen_losses": gen_losses,
                "critic_losses": critic_losses,
            },
            os.path.join(config.checkpoints_dir, "best_model.pth"),
        )

    if (epoch + 1) % config.save_after_every == 0:
        generator.eval()

        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "gen_optim_state_dict": gen_optim.state_dict(),
                "critic_optim_state_dict": critic_optim.state_dict(),
                "gen_losses": gen_losses,
                "critic_losses": critic_losses,
            },
            os.path.join(config.checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth"),
        )
