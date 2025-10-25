

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
    gap_length: int
    raw_norm_type: str  # 'group' or 'batch'
    spectral_norm_type: str  # 'batch', 'group', or 'none'
    fusion_norm_enabled: bool  # whether to add normalization before final layer

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

        # Build raw conv layers with configurable normalization
        raw_conv_layers: List[nn.Module] = []
        raw_conv_layers.extend([
            nn.Conv2d(
                in_channels=1,
                out_channels=cfg.n_filters[0],
                kernel_size=cfg.kernel_sizes[0],
                stride=cfg.strides[0],
                padding=cfg.paddings[0],
            ),
        ])
        
        if cfg.raw_norm_type == "group":
            raw_conv_layers.append(nn.GroupNorm(num_groups=min(cfg.n_filters[0], 4), num_channels=cfg.n_filters[0]))
        elif cfg.raw_norm_type == "batch":
            raw_conv_layers.append(nn.BatchNorm2d(cfg.n_filters[0]))
        else:
            raise ValueError(f"Unsupported raw_norm_type: {cfg.raw_norm_type}")
        
        raw_conv_layers.extend([act, nn.Dropout(cfg.raw_dropout_rate)])
        
        raw_conv_layers.append(
            nn.Conv2d(
                in_channels=cfg.n_filters[0],
                out_channels=cfg.n_filters[1],
                kernel_size=cfg.kernel_sizes[1],
                stride=cfg.strides[1],
                padding=cfg.paddings[1],
            )
        )
        
        if cfg.raw_norm_type == "group":
            raw_conv_layers.append(nn.GroupNorm(num_groups=min(cfg.n_filters[1], 4), num_channels=cfg.n_filters[1]))
        elif cfg.raw_norm_type == "batch":
            raw_conv_layers.append(nn.BatchNorm2d(cfg.n_filters[1]))
        
        raw_conv_layers.extend([act, nn.Dropout(cfg.raw_dropout_rate)])
        
        self.raw_conv = nn.Sequential(*raw_conv_layers)

        gap_size = (cfg.gap_length, cfg.gap_length)

        self.gap = nn.AdaptiveAvgPool2d(gap_size)

        # Build spectral network with configurable normalization
        spec_net_layers: List[nn.Module] = [nn.Linear(cfg.n_spectral_features, cfg.spectral_hidden_size)]
        
        if cfg.spectral_norm_type == "batch":
            spec_net_layers.append(nn.BatchNorm1d(cfg.spectral_hidden_size))
        elif cfg.spectral_norm_type == "group":
            spec_net_layers.append(nn.GroupNorm(num_groups=min(cfg.spectral_hidden_size, 4), num_channels=cfg.spectral_hidden_size))
        elif cfg.spectral_norm_type == "none":
            pass  # No normalization
        else:
            raise ValueError(f"Unsupported spectral_norm_type: {cfg.spectral_norm_type}")
        
        spec_net_layers.extend([act, nn.Dropout(cfg.spectral_dropout_rate)])
        
        self.spec_net = nn.Sequential(*spec_net_layers)

        # for (4,4), 32, 32 we would get here 32 * 4 * 4 = 512 + 32 = 544
        concatter_input_size = (cfg.n_filters[1] * gap_size[0] * gap_size[1]) + cfg.spectral_hidden_size

        # Build fusion head with optional normalization before final layer
        fusion_layers: List[nn.Module] = [
            nn.Linear(concatter_input_size, cfg.fusion_hidden_size),
            act,
            nn.Dropout(cfg.concat_dropout_rate),
        ]
        
        if cfg.fusion_norm_enabled:
            fusion_layers.append(nn.BatchNorm1d(cfg.fusion_hidden_size))
        
        fusion_layers.append(nn.Linear(cfg.fusion_hidden_size, 1))
        
        self.fusion = nn.Sequential(*fusion_layers)

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
