import albumentations as A
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate: float = 2e-4
batch_size: int = 16
num_workers: int = 2
image_size: int = 256
image_channels: int = 3
l1_lambda: float = 100.0
gp_lambda: float = 10.0
num_epochs: int = 500

load_model: bool = False
save_model: bool = True

checkpoint_generator: str = "generator.pth.tar"
checkpoint_discriminator: str = "discriminator.pth.tar"

both_transforms = A.Compose(
    [
        A.Resize(width=image_size, height=image_size),
        A.HorizontalFlip(p=0.5)
    ],
    additional_targets={"image0": "image"},
)

only_input_transforms = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        A.pytorch.ToTensorV2()
    ]
)

only_target_transforms = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        A.pytorch.ToTensorV2()
    ]
)
