import torch
from torch import nn
from typing import List, Tuple
from dataclasses import dataclass
from core.schemas import ModelConfig

@dataclass
class DeeperCustomConfig(ModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    paddings: List[Tuple[int, int]]
    activation: str

    model_name: str = "DeeperCustom"

    def __post_init__(self):
        assert len(self.n_filters) == 4, "n_filters must have length 4"
        assert len(self.kernel_sizes) == 4, "kernel_sizes must have length 4"
        assert len(self.strides) == 4, "strides must have length 4"
        assert len(self.paddings) == 4, "paddings must have length 4"

class DeeperCustom(nn.Module):
    """Simple 2D CNN for EEG data classification.
    This is thought for taking in data with a shape like
    (n_samples, n_channels, 1).
    """

    def __init__(
        self,
        cfg: DeeperCustomConfig,
    ):
        super(DeeperCustom, self).__init__()

        # Select activation
        if cfg.activation == "relu":
            act: nn.Module = nn.ReLU(inplace=True)
        elif cfg.activation == "leaky_relu":
            act = nn.LeakyReLU(inplace=True)
        elif cfg.activation == "gelu":
            act = nn.GELU()
        elif cfg.activation == "silu":
            act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {cfg.activation}")

        self.activation = act

        # Build layers in a DRY way
        layers = []
        in_channels = 1
        for i in range(4):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=cfg.n_filters[i],
                kernel_size=cfg.kernel_sizes[i],
                stride=cfg.strides[i],
                padding=cfg.paddings[i],
            )
            bn = nn.BatchNorm2d(cfg.n_filters[i])
            seq = nn.Sequential(conv, bn, nn.Dropout(cfg.dropout_rate), act)
            layers.append(seq)
            in_channels = cfg.n_filters[i]

        self.layers = nn.ModuleList(layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cfg.n_filters[3], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

@dataclass
class Deeper2DConfig(ModelConfig):
    n_filters: List[int]  # length 4
    kernel_sizes: List[Tuple[int, int]]  # length 4
    strides: List[Tuple[int, int]]  # length 4
    dropout_rate: float
    paddings: List[Tuple[int, int]]  # length 4

    model_name: str = "Deeper2D"

    def __post_init__(self):
        assert len(self.n_filters) == 4, "n_filters must have length 4"
        assert len(self.kernel_sizes) == 4, "kernel_sizes must have length 4"
        assert len(self.strides) == 4, "strides must have length 4"
        assert len(self.paddings) == 4, "paddings must have length 4"

class Deeper2D(nn.Module):
    """Simple 2D CNN for EEG data classification.
    This is thought for taking in data with a shape like
    (n_samples, n_channels, 1).
    """
    
    def __init__(
        self,
        cfg: Deeper2DConfig,
    ):
        super(Deeper2D, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = cfg.n_filters[0],
                kernel_size = cfg.kernel_sizes[0],
                stride = cfg.strides[0],
                padding = cfg.paddings[0],
            ),
            nn.BatchNorm2d(cfg.n_filters[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(
                in_channels = cfg.n_filters[0],
                out_channels = cfg.n_filters[1],
                kernel_size = cfg.kernel_sizes[1],
                stride = cfg.strides[1],
                padding = cfg.paddings[1],
            ),
            nn.BatchNorm2d(cfg.n_filters[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(
                in_channels = cfg.n_filters[1],
                out_channels = cfg.n_filters[2],
                kernel_size = cfg.kernel_sizes[2],
                stride = cfg.strides[2],
                padding = cfg.paddings[2],
            ),
            nn.BatchNorm2d(cfg.n_filters[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(
                in_channels = cfg.n_filters[2],
                out_channels = cfg.n_filters[3],
                kernel_size = cfg.kernel_sizes[3],
                stride = cfg.strides[3],
                padding = cfg.paddings[3],
            ),
            nn.BatchNorm2d(cfg.n_filters[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.n_filters[3], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


@dataclass
class Improved2DConfig(ModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    paddings: List[Tuple[int, int]]

    model_name: str = "Improved2D"

    def __post_init__(self):
        assert len(self.n_filters) == 3, "n_filters must have length 3"
        assert len(self.kernel_sizes) == 3, "kernel_sizes must have length 3"
        assert len(self.strides) == 3, "strides must have length 3"
        assert len(self.paddings) == 3, "paddings must have length 3"

class Improved2D(nn.Module):
    """Simple 2D CNN for EEG data classification.
    This is thought for taking in data with a shape like
    (n_samples, n_channels, 1).

    We will convolve over the samples and channels with 
    2d covolutions. We kind of understand a EEg samples as a
    very long image, with a single channel.
    """

    def __init__(
        self,
        cfg: Improved2DConfig,
    ):
        super(Improved2D, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = cfg.n_filters[0],
                kernel_size = cfg.kernel_sizes[0],
                stride = cfg.strides[0],
                padding = cfg.paddings[0],
            ),
            nn.BatchNorm2d(cfg.n_filters[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(
                in_channels = cfg.n_filters[0],
                out_channels = cfg.n_filters[1],
                kernel_size = cfg.kernel_sizes[1],
                stride = cfg.strides[1],
                padding = cfg.paddings[1],
            ),
            nn.BatchNorm2d(cfg.n_filters[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(
                in_channels = cfg.n_filters[1],
                out_channels = cfg.n_filters[2],
                kernel_size = cfg.kernel_sizes[2],
                stride = cfg.strides[2],
                padding = cfg.paddings[2],
            ),
            nn.BatchNorm2d(cfg.n_filters[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_rate),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.n_filters[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is (batch, channels=1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.l3(self.l2(self.l1(x)))
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

@dataclass
class Simple2D3LayersConfig(ModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float

    model_name: str = "Simple2D3Layers"

    def __post_init__(self):
        assert len(self.n_filters) == 3, "n_filters must have length 3"
        assert len(self.kernel_sizes) == 3, "kernel_sizes must have length 3"
        assert len(self.strides) == 3, "strides must have length 3"

class Simple2D3Layers(nn.Module):
    """Simple 2D CNN for EEG data classification.
    This is thought for taking in data with a shape like
    (n_samples, n_channels, 1).

    We will convolve over the samples and channels with 
    2d covolutions. We kind of understand a EEg samples as a
    very long image, with a single channel.
    """

    def __init__(
        self,
        cfg: Simple2D3LayersConfig, 
    ):
        super(Simple2D3Layers, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = cfg.n_filters[0],
                kernel_size = cfg.kernel_sizes[0],
                stride = cfg.strides[0],
                padding = (0,0),
            ),
            nn.BatchNorm2d(cfg.n_filters[0]),
            nn.Dropout(cfg.dropout_rate),
            nn.ReLU(inplace=True),
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(
                in_channels = cfg.n_filters[0],
                out_channels = cfg.n_filters[1],
                kernel_size = cfg.kernel_sizes[1],
                stride = cfg.strides[1],
                padding = (0,0),
            ),
            nn.BatchNorm2d(cfg.n_filters[1]),
            nn.Dropout(cfg.dropout_rate),
            nn.ReLU(inplace=True),
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(
                in_channels = cfg.n_filters[1],
                out_channels = cfg.n_filters[2],
                kernel_size = cfg.kernel_sizes[2],
                stride = cfg.strides[2],
                padding = (0,0),
            ),
            nn.BatchNorm2d(cfg.n_filters[2]),
            nn.Dropout(cfg.dropout_rate),
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.n_filters[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is (batch, channels=1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.l3(self.l2(self.l1(x)))
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

