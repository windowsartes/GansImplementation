import torch

from DCGAN.models import Generator, Discriminator


def test_discriminator_output_shape():
    N: int = 8
    in_channels: int = 3
    H: int = 64
    W: int = 64

    input_tensor: torch.Tensor = torch.randn((N, in_channels, H, W))

    model: Discriminator = Discriminator(in_channels, 8)

    assert model(input_tensor).shape == (N, 1, 1, 1)

def test_generator_output_shape():
    N: int = 8
    in_channels: int = 3
    H: int = 64
    W: int = 64

    z_dim: int = 100

    model: Generator = Generator(z_dim, in_channels, 8) 

    input_tensor: torch.Tensor = torch.randn((N, z_dim, 1, 1))

    assert model(input_tensor).shape == (N, in_channels, H, W)
