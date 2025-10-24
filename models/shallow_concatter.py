

from typing import List, Tuple

from torch import nn
import torch
from core.schemas import NetworkConfig


class ShallowerConcatterConfig(NetworkConfig):
    model_name: str = "ShallowerConcatter"
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    raw_dropout_rate: float
    spectral_dropout_rate: float
    paddings: List[Tuple[int, int]]
    activation: str
    n_spectral_features: int
    spectral_hidden_size: int
    concat_dropout_rate: float
    fusion_hidden_size: int

class ShallowerConcatter(nn.Module):
    def __init__(
        self,
        cfg: ShallowerConcatterConfig,
    ):
        super().__init__()

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

        self.raw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=cfg.n_filters[0],
                kernel_size=cfg.kernel_sizes[0],
                stride=cfg.strides[0],
                padding=cfg.paddings[0],
            ),
            nn.GroupNorm(num_groups=min(cfg.n_filters[0], 4), num_channels=cfg.n_filters[0]),
            act,
            nn.Dropout(cfg.raw_dropout_rate),
            nn.Conv2d(
                in_channels=cfg.n_filters[0],
                out_channels=cfg.n_filters[1],
                kernel_size=cfg.kernel_sizes[1],
                stride=cfg.strides[1],
                padding=cfg.paddings[1],
            ),
            nn.GroupNorm(num_groups=min(cfg.n_filters[1], 4), num_channels=cfg.n_filters[1]),
            act,
            nn.Dropout(cfg.raw_dropout_rate),
        )

        gap_size = (4, 4)

        self.gap = nn.AdaptiveAvgPool2d(gap_size)

        self.spec_net = nn.Sequential(
            nn.Linear(cfg.n_spectral_features, cfg.spectral_hidden_size),
            nn.BatchNorm1d(cfg.spectral_hidden_size),
            act,
            nn.Dropout(cfg.spectral_dropout_rate),
        )

        # for (4,4), 32, 32 we would get here 32 * 4 * 4 = 512 + 32 = 544
        concatter_input_size = (cfg.n_filters[1] * gap_size[0] * gap_size[1]) + cfg.spectral_hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(concatter_input_size, cfg.fusion_hidden_size),
            act,
            nn.Dropout(cfg.concat_dropout_rate),
            nn.Linear(cfg.fusion_hidden_size, 1),
        )

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

        x_raw = self.raw_conv(x_raw)

        x_raw = self.gap(x_raw)
        x_raw = x_raw.flatten(1)

        x_spectral = self.spec_net(x_spectral)

        return self.fusion(torch.cat([x_raw, x_spectral], dim=1))
