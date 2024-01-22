import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import Pix2Pix.config as config
from Pix2Pix.dataset import Pix2PixDataset
from Pix2Pix.models import Generator, Discriminator
from Pix2Pix.utils import save_checkpoint, load_checkpoint, save_some_examples


def train_function(
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        discriminator: Discriminator,
        generator: Generator,
        optimizer_discriminator: optim.Optimizer,
        optimizer_generator: optim.Optimizer,
        bce_loss: nn.BCEWithLogitsLoss,
        l1_loss: nn.L1Loss,
        scaler_discriminator: torch.cuda.amp.GradScaler,
        scaler_generator: torch.cuda.amp.GradScaler
    ) -> None:
    """
    Function to train your Pix2Pix model; Please notice that this function uses mixed precision training;

    Args:
        dataloader (DataLoader[tuple[torch.Tensor, torch.Tensor]]): dataloader which will provide model's input;
        discriminator (Discriminator): Discriminator model;
        generator (Generator): Generator model;
        optimizer_discriminator (optim.Optimizer): Discriminator' optimizer;
        optimizer_generator (optim.Optimizer): Generator's optimizer;
        bce_loss (nn.BCEWithLogitsLoss): BCE Loss instance;
        l1_loss (nn.L1Loss): L1 loss instance;
        scaler_discriminator (torch.cuda.amp.GradScaler): Scaler for discriminator
        scaler_generator (torch.cuda.amp.GradScaler): Scaler for generator;
    """
    loop = tqdm(dataloader, leave=True)

    for index, (input_image, target_image) in enumerate(loop):
        input_image, target_image = input_image.to(config.device), target_image.to(config.device)

        # train discriminator
        with torch.cuda.amp.autocast():
            generator_output: torch.Tensor = generator(input_image)

            discriminator_output_real = discriminator(input_image, target_image)
            discriminator_output_fake = discriminator(input_image, generator_output)

            discriminator_loss_real = bce_loss(discriminator_output_real,
                torch.ones_like(discriminator_output_real)) 
            discriminator_loss_fake = bce_loss(discriminator_output_fake,
                torch.zeros_like(discriminator_output_fake))

            discriminator_loss = (discriminator_loss_fake + discriminator_loss_real) / 2

        discriminator.zero_grad()
        scaler_discriminator.scale(discriminator_loss).backward(retain_graph=True)
        scaler_discriminator.step(optimizer_discriminator)
        scaler_discriminator.update()

        with torch.cuda.amp.autocast():
            generator_output_loss = bce_loss(discriminator_output_fake,
                torch.ones_like(discriminator_output_fake))
            additional_loss = l1_loss(discriminator_output_fake, target_image) * config.l1_lambda
            generator_loss = generator_output_loss + additional_loss

        generator.zero_grad()
        scaler_generator.scale(generator_loss).backward(retain_graph=True)
        scaler_generator.step(optimizer_generator)
        scaler_generator.update()


if __name__ == "__main__":
    discriminator: Discriminator = Discriminator(3, 3, (64, 128, 256, 512)).to(config.device)
    optimizer_discriminator: optim.Optimizer = optim.Adam(discriminator.parameters(),
        lr=config.learning_rate, betas=(0.5, 0.999))
    scaler_discriminator: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler()
    
    generator: Generator = Generator(3, 64).to(config.device)
    optimizer_generator: optim.Optimizer = optim.Adam(generator.parameters(),
        lr=config.learning_rate, betas=(0.5, 0.999))
    scaler_generator: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler()

    bce_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    l1_loss: nn.L1Loss = nn.L1Loss()

    if config.load_model:
        load_checkpoint(config.checkpoint_discriminator, discriminator, optimizer_discriminator,
            config.learning_rate)
        load_checkpoint(config.checkpoint_generator, generator, optimizer_generator,
            config.learning_rate)

    train_dataset: Pix2PixDataset = Pix2PixDataset(os.path.join("data", os.path.join("maps", "train")))
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(train_dataset,
        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_dataset: Pix2PixDataset = Pix2PixDataset(os.path.join("data", os.path.join("maps", "val")))
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(val_dataset,
        batch_size=config.batch_size, shuffle=False)

    for epoch in range(config.num_epochs):
        train_function(train_dataloader, discriminator, generator, optimizer_discriminator, 
        optimizer_generator, bce_loss, l1_loss, scaler_discriminator, scaler_generator)

        if config.save_model and not epoch % 10:
            save_checkpoint(discriminator, optimizer_discriminator, config.checkpoint_discriminator)
            save_checkpoint(generator, optimizer_generator, config.checkpoint_generator)
        
        save_some_examples(generator, val_dataloader, epoch, "evaluation")
