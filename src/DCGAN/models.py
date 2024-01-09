import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_image: int, features_d: int) -> None:
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels_image, features_d, kernel_size=4, stride=2, padding=1), # 64x64 ->x32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), # 32x32 -> 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1), # 16x16 -> 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1), # 8x8 -> 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int, padding: int) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.discriminator(input)
    

class Generator(nn.Module):
    def __init__(self, z_dim: int, channels_image: int, features_g: int):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0), # z_dimx1x1 -> features_g*16x4x4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 4x4 -> 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 8x8 -> 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 16x16 -> 32x32
            nn.ConvTranspose2d(features_g*2, channels_image,
                kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            nn.Tanh(), 
        )

    def _block(self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int, padding: int) -> nn.Sequential:

        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.generator(input)


def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
