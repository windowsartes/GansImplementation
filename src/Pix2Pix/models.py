import torch
import torch.nn as nn


class DiscriminatorCNNBlock(nn.Module):
    """
    CNN block used by Discriminator; contains Conv2d layer, InstanceNorm and LeakyReLu activation;
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False,
                padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


class Discriminator(nn.Module):
    """
    Discriminator model used in Pix2Pix architecture;
    """
    def __init__(self, in_channels: int, out_channels: int,
        features: tuple[int, ...] = (64, 128, 256, 512)):
        super().__init__()

        self.initial_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, features[0], kernel_size=4, stride=2,
                padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)            
        )

        layers: list[nn.Module] = []

        in_channels = features[0]

        for feature in features[1:-1]:
            layers.append(
                DiscriminatorCNNBlock(in_channels, feature, 2)
            )
            in_channels = feature     

        layers.append(
            DiscriminatorCNNBlock(in_channels, features[-1], 1)
        )
        
        layers.append(
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model: nn.Sequential = nn.Sequential(*layers)


    def forward(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        input = torch.cat([input, output], dim=1)
        input = self.initial_block(input)

        return self.model(input)


class GeneratorCNNBlock(nn.Module):
    """
    CNN block used by Generator. In 'down' mode contain Conv layer, InstanceNorm and RelU or LeakyReLu activation;
    In 'up' mode contains ConvTransposed layer, InstanceNorm and the same activation option;
    """
    def __init__(self, in_channels: int, out_channels: int, down: bool = True,
        activation: str = "relu", use_dropout: bool = True):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down
                else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)
        )

        self.use_dropout: bool = use_dropout
        self.dropout_layer: nn.Dropout = nn.Dropout(0.5)  

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.block(input)

        if self.use_dropout:
            return self.dropout_layer(input)
        return input


class Generator(nn.Module):
    """
    Generator model used in Pix2Pix architecture;
    """
    def __init__(self, in_channels: int, features_g: int):
        super().__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features_g, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )  # 256 -> 128

        self.down_1 = GeneratorCNNBlock(features_g, features_g * 2, down=True, activation="leaky",
            use_dropout=False)  # 128 -> 64
        self.down_2 = GeneratorCNNBlock(features_g * 2, features_g * 4, down=True, activation="leaky",
            use_dropout=False)  # 64 -> 32
        self.down_3 = GeneratorCNNBlock(features_g * 4, features_g * 8, down=True, activation="leaky",
            use_dropout=False)  # 32 -> 16
        self.down_4 = GeneratorCNNBlock(features_g * 8, features_g * 8, down=True, activation="leaky",
            use_dropout=False)  # 16 -> 8
        self.down_5 = GeneratorCNNBlock(features_g * 8, features_g * 8, down=True, activation="leaky",
            use_dropout=False)  # 8 -> 4
        self.down_6 = GeneratorCNNBlock(features_g * 8, features_g * 8, down=True, activation="leaky",
            use_dropout=False)  # 4 -> 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features_g * 8, features_g * 8, 4, 2, 1, padding_mode="reflect"),  # 2 -> 1
            nn.ReLU()
        )

        self.up_1 = GeneratorCNNBlock(features_g * 8, features_g * 8, down=False, activation="relu",
            use_dropout=True)  # 1 -> 2
        self.up_2 = GeneratorCNNBlock(features_g * 16, features_g * 8, down=False, activation="relu",
            use_dropout=True)  # 2 -> 4
        self.up_3 = GeneratorCNNBlock(features_g * 16, features_g * 8, down=False, activation="relu",
            use_dropout=True)  # 4 -> 8
        self.up_4 = GeneratorCNNBlock(features_g * 16, features_g * 8, down=False, activation="relu",
            use_dropout=False)  # 8 -> 16
        self.up_5 = GeneratorCNNBlock(features_g * 16, features_g * 4, down=False, activation="relu",
            use_dropout=False)  # 16 -> 32
        self.up_6 = GeneratorCNNBlock(features_g * 8, features_g * 2, down=False, activation="relu",
            use_dropout=False)  # 32 -> 64
        self.up_7 = GeneratorCNNBlock(features_g * 4, features_g, down=False, activation="relu",
            use_dropout=False)  # 64 -> 128

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(features_g * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() 
        )  # 128 -> 256

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        initial_output = self.initial_down(input)
        down_1_output = self.down_1(initial_output)
        down_2_output = self.down_2(down_1_output)
        down_3_output = self.down_3(down_2_output)
        down_4_output = self.down_4(down_3_output)
        down_5_output = self.down_5(down_4_output)
        down_6_output = self.down_6(down_5_output)

        bottleneck_output = self.bottleneck(down_6_output)

        up_1_output = self.up_1(bottleneck_output)
        up_2_output = self.up_2(torch.cat([up_1_output, down_6_output], dim=1))
        up_3_output = self.up_3(torch.cat([up_2_output, down_5_output], dim=1))
        up_4_output = self.up_4(torch.cat([up_3_output, down_4_output], dim=1))
        up_5_output = self.up_5(torch.cat([up_4_output, down_3_output], dim=1))
        up_6_output = self.up_6(torch.cat([up_5_output, down_2_output], dim=1))
        up_7_output = self.up_7(torch.cat([up_6_output, down_1_output], dim=1))

        return self.final_layer(torch.cat([up_7_output, initial_output], dim=1))
