import torch
from torch import nn
from einops import rearrange
from typing import Tuple


class Generator(nn.Module):
    def __init__(
        self,
        img_channels: int,
        latent_dim: int,
        base_features: int,
        num_blocks: int,
        num_classes: int,
        use_upsample: bool,
    ) -> None:
        super().__init__()

        self.use_upsample = use_upsample
        self.initial_layer_out_channels = base_features * (2**num_blocks)

        self.embed = nn.Embedding(num_classes, latent_dim)
        self.net = nn.Sequential(
            self._block(
                latent_dim * 2,
                self.initial_layer_out_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                first_block=True,
            ),
            self._make_block_chain(
                self.initial_layer_out_channels, base_features, num_blocks
            ),
            nn.ConvTranspose2d(
                in_channels=base_features,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(labels)
        embedding = rearrange(embedding, "b w -> b w 1 1")

        x = torch.cat([x, embedding], dim=1)

        return self.net(x)

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        upsample: bool = True,
        first_block: bool = False,
    ) -> nn.Sequential:
        layers = []

        if self.use_upsample and not first_block:
            if upsample:
                layers.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )

            layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                ),
            )
        else:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                )
            )

        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_block_chain(
        self, initial_channels: int, gen_base_features: int, num_blocks: int
    ) -> nn.Sequential:
        layers = []
        in_channels = initial_channels

        for i in range(num_blocks):
            out_channels = gen_base_features * 2 ** (num_blocks - i - 1)
            layers.append(
                self._block(
                    in_channels, out_channels, kernel_size=3, padding=1, stride=1
                )
            )
            in_channels = out_channels

        return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(
        self,
        img_channels: int,
        base_features: int,
        num_blocks: int,
        num_classes: int,
        img_size: int | Tuple[int, int],
        alpha: float = 0.2,
    ) -> None:
        super().__init__()

        self.alpha = alpha

        if isinstance(img_size, Tuple):
            self.img_h = img_size[0]
            self.img_w = img_size[1]
        elif isinstance(img_size, int):
            self.img_h = img_size
            self.img_w = img_size
        else:
            raise Exception("`img_size` should be of type `int` or `Tuple[int, int]`")

        self.embed = nn.Embedding(num_classes, self.img_h * self.img_w)
        self.net = nn.Sequential(
            self._block(
                img_channels + 1,
                base_features,
                kernel_size=4,
                stride=2,
                padding=1,
                use_norm=False,
            ),
            self._make_block_chain(base_features, num_blocks),
            nn.Conv2d(
                in_channels=base_features * 2**num_blocks,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(labels)
        embedding = rearrange(
            embedding, "b (h w) -> b 1 h w", h=self.img_h, w=self.img_w
        )

        x = torch.cat([x, embedding], dim=1)

        return self.net(x)

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_norm: bool = True,
    ) -> nn.Sequential:
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        layers.append(nn.LeakyReLU(self.alpha))
        return nn.Sequential(*layers)

    def _make_block_chain(
        self, initial_in_channels: int, num_blocks: int
    ) -> nn.Sequential:
        layers = []
        in_channels = initial_in_channels

        for _ in range(num_blocks):
            layers.append(
                self._block(
                    in_channels,
                    in_channels * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    use_norm=True,
                )
            )

            in_channels = in_channels * 2

        return nn.Sequential(*layers)


def test():
    noise = torch.randn(1, 100, 1, 1)
    labels = torch.LongTensor([1])
    generator = Generator(1, 100, 128, 4, 4, True)
    fake = generator(noise, labels)
    print(fake.shape)


if __name__ == "__main__":
    test()
