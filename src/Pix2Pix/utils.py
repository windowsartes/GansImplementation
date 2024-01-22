import os

import torch
from torchvision.utils import save_image

import Pix2Pix.config as config


def save_some_examples(generator: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epoch: int, 
    folder: str) -> None:
    """
    Using this function you can verify given model's output by saving it's generation results to the given file;

    Args:
        generator (torch.nn.Module): model which output you want to save;
        val_dataloader (torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]): dataloader
        you want to use to get input data;
        epoch (int): current epoch that will be used in saved files name so we can see output quality dynamically;
        folder (str): folder where you want to save model's output
    """
    
    input_image, target_image = next(iter(val_dataloader))
    input_image, target_image = input_image.to(config.device), target_image.to(config.device)

    generator.eval()
    with torch.no_grad():
        target_image_fake = generator(input_image)

        save_image(target_image_fake * 0.5 + 0.5, os.path.join(folder, f"output_{epoch}.png"))
        save_image(input_image * 0.5 + 0.5, os.path.join(folder, f"input_{epoch}.png"))
        save_image(target_image * 0.5 + 0.5, os.path.join(folder, f"answer_{epoch}.png"))
    generator.train()

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> None:
    """
    This function helps to store model and optimizer to you can continue training later;

    Args:
        model (torch.nn.Module): model you want to store;
        optimizer (torch.optim.Optimizer): optimizer which state you want to store;
        filename (str): path to file where you want to store model and optimizer; usually, a tar archive;
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    learning_rate: float) -> None:
    """
    Function that loads model and optimizer from given checkpoint so you can continue training;

    Args:
        checkpoint_file_path (str): path to checkpoint you want to load;
        model (torch.nn.Module): model which weights you want to load;
        optimizer (torch.optim.Optimizer): optimizer which state you want to load;
        learning_rate (float): learning rate you want to use in loaded optimizer;
    """

    checkpoint = torch.load(checkpoint_file_path, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
