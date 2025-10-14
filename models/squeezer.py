from __future__ import annotations
from typing import List
from typing import Tuple

from pydantic import model_validator
from torch import nn
import torch

from core import validate_kernel
from core.schemas import NetworkConfig


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(SEBlock, self).__init__()
        bottleneck_channels = max(in_channels // reduction_ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, bottleneck_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class DeeperSEConfig(NetworkConfig):
    model_name: str = "DeeperSE"
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    paddings: List[Tuple[int, int]]
    activation: str
    reduction_ratio: int
    input_shape: Tuple[int, int] = (1000, 68)

    @model_validator(mode="after")
    def validate_reduction_ratio(self):
        assert self.reduction_ratio > 0, "reduction_ratio must be positive"
        return self

    @model_validator(mode="after")
    def validate_kernel_params(self):
        if not validate_kernel(self.kernel_sizes, self.strides, self.paddings, self.input_shape):
            raise ValueError("Invalid kernel parameters")
        return self 

class DeeperSE(nn.Module):
    def __init__(self, cfg: DeeperSEConfig):
        super(DeeperSE, self).__init__()

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

        layers = []
        in_channels = 1
        for i in range(len(cfg.n_filters)):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, cfg.n_filters[i], cfg.kernel_sizes[i], cfg.strides[i], cfg.paddings[i]),
                nn.BatchNorm2d(cfg.n_filters[i]),
                act,
                SEBlock(cfg.n_filters[i], cfg.reduction_ratio),
                nn.Dropout(cfg.dropout_rate),
            ))
            in_channels = cfg.n_filters[i]

        self.layers = nn.Sequential(*layers)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cfg.n_filters[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.layers(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)