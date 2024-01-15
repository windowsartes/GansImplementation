import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from WGAN_GP.models import Critic, Generator, initialize_weights

def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: torch.device):
    batch_size, channels, h, w = real.shape
    epsilon: torch.Tensor = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, h, w).to(device)

    interpolated_images: torch.Tensor = epsilon * real + (1 - epsilon) * fake

    # value inside the penalty term
    mixed_scores: torch.Tensor = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)

    return gradient_penalty


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate: float = 1e-4

batch_size: int = 64
image_size: int = 64
channels_image: int = 3

z_dim: int = 100
features_critic: int = 64
features_generator: int = features_critic

num_epochs: int = 10

critic_iterations: int = 5
lambda_penalty: float = 10.0

transformations = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_image)], [0.5 for _ in range(channels_image)]
        )    
    ]
)

dir_path: os.PathLike[str] = Path(os.path.dirname(os.path.realpath(__file__)))

dataset = datasets.ImageFolder(root="celeb_dataset", transform=transformations)

dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dataset,
    batch_size=batch_size, shuffle=True)

generator = Generator(z_dim, channels_image, features_generator).to(device)
initialize_weights(generator)
optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

critic = Critic(channels_image, features_critic).to(device)
initialize_weights(critic)
optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))

fixed_noise: torch.Tensor = torch.randn((32, z_dim, 1, 1)).to(device)

generator.train()
critic.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)

        # train critic
        for _ in range(critic_iterations):
            noise: torch.Tensor = torch.randn((batch_size, z_dim, 1, 1)).to(device)

            fake = generator(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            penalty = gradient_penalty(critic, real, fake, device=device)

            loss_critic = torch.mean(critic_fake) - torch.mean(critic_real) + \
                lambda_penalty * penalty

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()

        output = critic(fake).reshape(-1)
        # train generator: we want to maximize E[Critic(fake)] <-> minimize -1 * E[Critic(fake)]
        loss_generator = -torch.mean(output)

        generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_critic:.4f}, " + 
                      f"loss G: {loss_generator:.4f}")
                      