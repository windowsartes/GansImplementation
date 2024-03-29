import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from DCGAN.models import Discriminator, Generator, initialize_weights


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate: float = 2e-4

batch_size: int = 32
image_size: int = 64
channels_image: int = 1

z_dim: int = 100
features_discriminator: int = 64
features_generator: int = features_discriminator

num_epochs: int = 100

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

dataset = datasets.MNIST(root=str(Path.joinpath(dir_path, "dataset")), train=True,  # type: ignore
    transform=transformations, download=True)
dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dataset,
    batch_size=batch_size, shuffle=True)

generator = Generator(z_dim, channels_image, features_generator).to(device)
initialize_weights(generator)
optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

discriminator = Discriminator(channels_image, features_discriminator).to(device)
initialize_weights(discriminator)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

criterion = nn.BCELoss()

fixed_noise: torch.Tensor = torch.randn((32, z_dim, 1, 1)).to(device)

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise: torch.Tensor = torch.randn((batch_size, z_dim, 1, 1)).to(device)

        fake = generator(noise)

        # train discriminator: max[log(D(x)) + log(1 - D(G(z)))]
        discriminator_real = discriminator(real).reshape(-1)
        loss_discriminator_real = criterion(discriminator_real, torch.ones_like(discriminator_real))

        discriminator_fake = discriminator(fake).reshape(-1)  
        loss_discriminator_fake = criterion(discriminator_fake, torch.zeros_like(discriminator_fake))

        discriminator.zero_grad()

        loss_discriminator = (loss_discriminator_fake + loss_discriminator_real) / 2 
        loss_discriminator.backward(retain_graph=True)
        optimizer_discriminator.step()

        # train generator: min[log(1 - D(G(z)))] <-> max[log(D(G(z)))]
        output = discriminator(fake).reshape(-1)
        loss_generator = criterion(output, torch.ones_like(output))

        generator.zero_grad()

        loss_generator.backward()
        optimizer_generator.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss D: {loss_discriminator:.4f}, " + 
                      f"loss G: {loss_generator:.4f}")
