import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from WGAN.models import Critic, Generator, initialize_weights


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate: float = 5e-5

batch_size: int = 64
image_size: int = 64
channels_image: int = 3

z_dim: int = 100
features_critic: int = 64
features_generator: int = features_critic

num_epochs: int = 10

critic_iterations: int = 5
weight_clip: float = 0.01

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
optimizer_generator = optim.RMSprop(generator.parameters(), lr=learning_rate)

critic = Critic(channels_image, features_critic).to(device)
initialize_weights(critic)
optimizer_critic = optim.RMSprop(critic.parameters(), lr=learning_rate)

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

            # minus because the critic wants to maximize value inside brackets
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()

            for parameter in critic.parameters():
                parameter.data.clamp_(-weight_clip, weight_clip)

        output = critic(fake).reshape(-1)
        # train generator: we want to maximize E[Critic(fake)] <-> minimize -1 * E[Critic(fake)]
        loss_generator = -torch.mean(output)

        generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_critic:.4f}, " + 
                      f"loss G: {loss_generator:.4f}")
