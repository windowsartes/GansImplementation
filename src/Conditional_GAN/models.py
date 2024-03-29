import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Critic model. Based on DCGAN, except last layer: critic model doesn't have last activation layer;
    """
    def __init__(self, channels_image: int, features_d: int, num_classes: int, image_size: int) -> None:
        super(Critic, self).__init__()

        self.image_size: int = image_size
        
        self.critic = nn.Sequential(
            nn.Conv2d(channels_image + 1, features_d, kernel_size=4, stride=2, padding=1),  # 64x64 ->x32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 32x32 -> 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16 -> 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8 -> 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0)
        )

        self.embedding = nn.Embedding(num_classes, 1 * image_size * image_size)

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
            nn.InstanceNorm2d(num_features=out_channels, affine=True),  # in the paper they used LayerNorm
            nn.LeakyReLU(0.2),
        )

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # We'll interpret this like an additional channel  
        embedding: torch.Tensor = self.embedding(label).view(label.shape[0], 1, self.image_size,
            self.image_size)
        input = torch.cat([input, embedding], dim=1)

        return self.critic(input)
    

class Generator(nn.Module):
    """
    Generator model. Based on DCGAN;
    """
    def __init__(self, channels_noise: int, channels_image: int, features_g: int, num_classes: int,
        image_size: int, embedding_size: int):
        super(Generator, self).__init__()

        self.image_size: int = image_size

        self.generator = nn.Sequential(
            self._block(channels_noise + embedding_size, features_g * 16, 4, 1, 0),  # z_dimx1x1 -> features_g*16x4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 4x4 -> 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 8x8 -> 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 16x16 -> 32x32
            nn.ConvTranspose2d(features_g * 2, channels_image,
                kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Tanh(), 
        )

        self.embedding = nn.Embedding(num_classes, embedding_size)

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

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # before we had only a noise vector z, since it's random the input will be different so
        # model' output will also be different.
        # but for now we'll also generate a vector from this  embedding layer, so it contains some
        # information about labels and this embedding we'll interpret like new channel for input  
        embedding = self.embedding(label).unsqueeze(2).unsqueeze(3)
        # print(embedding.shape, input.shape, label.shape, self.embedding(label).shape)
        input = torch.cat([input, embedding], dim=1)

        return self.generator(input)


def initialize_weights(model: nn.Module) -> None:
    """
    Weights initialization, proposed in DCGAN paper;

    Args:
        model (nn.Module): model which weights you want to initialize;
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
