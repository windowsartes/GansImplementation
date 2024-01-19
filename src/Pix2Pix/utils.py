import os
import typing as tp

import torch
from torchvision.utils import save_image

import Pix2Pix.config as config


def save_some_examples(generator: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epoch: int, 
    folder: str) -> None:
    
    input_image, target_image = next(iter(val_dataloader))
    input_image, target_image = input_image.to(config.device), target_image.to(config.device)

    generator.eval()
    with torch.no_grad():
        target_image_fake = generator(input_image)

        save_image(target_image_fake * 0.5 + 0.5, os.path.join(folder, f"output_{epoch}.png"))
        save_image(input_image*0.5 + 0.5, os.path.join(folder, f"input_{epoch}.png"))
        save_image(target_image*0.5 + 0.5, os.path.join(folder, f"answer_{epoch}.png"))
    generator.train()

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    learning_rate: float) -> None:
    checkpoint = torch.load(checkpoint_file_path, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
