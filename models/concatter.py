import torch
from torch import nn
from typing import List, Tuple

from core.schemas import NetworkConfig

class ConcatterConfig(NetworkConfig):
    model_name: str = "Concatter"
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    paddings: List[Tuple[int, int]]
    activation: str
    n_spectral_features: int

class Concatter(nn.Module):
    """
    This network consists of two branches plus a concattenation head.
    Raw branch:
        - 4 conv layers in the style of deeper custom.
        - each layer: (conv, bn, act, dropout)
    Spectral branch:
        - Right now, literally nothing.
    Concattenation head:
        - we pool the output of the cnns
        - we concatenate it to the spectral features
        - single perceptron for output
    """

    def __init__(
        self,
        cfg: ConcatterConfig,
    ):
        super().__init__()

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

        # Build convolutional layers
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

        # The input to the classifier is conv_out + n_spectral_features
        self.classifier = nn.Linear(cfg.n_filters[3] + cfg.n_spectral_features, 1)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        x: Tuple[torch.Tensor, torch.Tensor], shape (batch, channels, time) or (batch, 1, channels, time)
        spectral_features: shape (batch, n_spectral_features)
        """
        x_raw, x_spectral = x
        if x_raw.dim() == 3:
            x_raw = x_raw.unsqueeze(1)
        for layer in self.layers:
            x_raw = layer(x_raw)
        x_raw = self.global_avg_pool(x_raw)
        x_raw = x_raw.flatten(1)
        x_combined = torch.cat([x_raw, x_spectral], dim=1)
        return self.classifier(x_combined)