import torch

from Pix2Pix.models import Discriminator, Generator


def test_discriminator_output_shape():
    N: int = 8
    in_channels: int = 3
    out_channels: int = 3
    H: int = 256
    W: int = 256

    input_tensor: torch.Tensor = torch.randn((1, in_channels, 256, 256))
    output_tensor: torch.Tensor = torch.randn((1, out_channels, 256, 256))

    model: Discriminator = Discriminator(in_channels, out_channels, (64, 128, 256, 512))

    assert model(input_tensor, output_tensor).shape == (1, 1, 30, 30)

def test_generator_output_shape():
    input_tensor = torch.randn((1, 3, 256, 256))

    model: Generator = Generator(3, 64)

    assert model(input_tensor).shape == (1, 3, 256, 256)
