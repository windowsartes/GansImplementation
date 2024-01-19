import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import Pix2Pix.config as config


class MapDataset(Dataset): # type: ignore
    def __init__(self, path_to_data: str):
        self.path_to_data: str = path_to_data
        self.list_files: list[str] = os.listdir(self.path_to_data)

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_file: str = self.list_files[index]
        imaga_path: str = os.path.join(self.path_to_data, image_file)

        image: np.typing.NDArray[np.int64] = np.array(Image.open(imaga_path))

        image_width: int = image.shape[1]
        
        image_input: np.typing.NDArray[np.int64] = image[:, :600, :]
        image_target: np.typing.NDArray[np.int64] = image[:, 600:, :]

        augmentations: dict[str, torch.Tensor] = config.both_transforms(image=image_input, image0=image_target)

        image_input_tensor, image_target_tensor = augmentations["image"], augmentations["image0"]

        image_input_tensor = config.only_input_transforms(image=image_input_tensor)["image"]
        image_target_tensor = config.only_target_transforms(image=image_target_tensor)["image"]

        return image_input_tensor, image_target_tensor
