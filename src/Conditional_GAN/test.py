import pytest
import torch

from Conditional_GAN.models import Generator, Critic


def test_critic_output_shape():
    N: int = 8
    in_channels: int = 3
    H: int = 64
    W: int = 64

    num_classes: int = 10

    input_tensor: torch.Tensor = torch.randn((N, in_channels, H, W))

    model: Critic = Critic(in_channels, 8, num_classes, H)

    assert model(input_tensor, torch.ones((N,)).type(torch.LongTensor)).shape == (N, 1, 1, 1)

testdata: list[int] = [1, 10, 100, 1000]
@pytest.mark.parametrize("embedding_size", testdata)
def test_generator_output_shape(embedding_size: int):
    N: int = 8
    in_channels: int = 3
    H: int = 64
    W: int = 64

    z_dim: int = 100

    num_classes: int = 10

    model: Generator = Generator(z_dim, in_channels, 8, num_classes, H, embedding_size)

    input_tensor: torch.Tensor = torch.randn((N, z_dim, 1, 1))

    assert model(input_tensor, torch.ones((N,)).type(torch.LongTensor)).shape == (N, in_channels, H, W)
