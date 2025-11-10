from typing import List, Tuple

from torch import nn
import torch
from pydantic import model_validator

from core.schemas import NetworkConfig
from models.squeezer import SEBlock

# Architecture presets for 2-layer shallow architectures
# Designed for input shape: (batch, 1, 68 channels, 1000 time_steps)
ARCHITECTURE_PRESETS = {
    "proven_tiny": {
        "n_filters": [16, 32],
        "kernel_sizes": [(40, 2), (8, 5)],
        "strides": [(12, 6), (10, 5)],
        "paddings": [(5, 0), (1, 1)],
    },
    "tiny_2layer": {
        "n_filters": [16, 32],
        "kernel_sizes": [(25, 5), (5, 5)],
        "strides": [(8, 4), (4, 4)],
        "paddings": [(2, 2), (1, 1)],
    },
    "small_2layer": {
        "n_filters": [32, 64],
        "kernel_sizes": [(30, 7), (6, 5)],
        "strides": [(6, 3), (3, 3)],
        "paddings": [(2, 2), (1, 1)],
    },
    "compact_2layer": {
        "n_filters": [16, 32],
        "kernel_sizes": [(20, 5), (5, 3)],
        "strides": [(10, 5), (5, 3)],
        "paddings": [(2, 1), (1, 1)],
    },
    "medium_2layer": {
        "n_filters": [48, 96],
        "kernel_sizes": [(35, 9), (7, 5)],
        "strides": [(5, 3), (3, 2)],
        "paddings": [(3, 2), (2, 1)],
    },
    "wide_2layer": {
        "n_filters": [64, 128],
        "kernel_sizes": [(40, 11), (8, 7)],
        "strides": [(4, 3), (2, 2)],
        "paddings": [(3, 3), (2, 2)],
    },
}


def get_architecture_preset(preset_name: str) -> dict:
    """Get architecture parameters from preset name."""
    if preset_name not in ARCHITECTURE_PRESETS:
        raise ValueError(
            f"Unknown architecture preset: {preset_name}. "
            f"Available: {list(ARCHITECTURE_PRESETS.keys())}"
        )
    return ARCHITECTURE_PRESETS[preset_name]


class ShallowConcatterSEConfig(NetworkConfig):
    model_name: str = "ShallowConcatterSE"
    
    # SE block parameters
    use_se_blocks: bool
    reduction_ratio: int
    
    # Raw branch (conv) parameters
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    paddings: List[Tuple[int, int]]
    raw_norm_type: str  # 'batch' or 'group'
    raw_dropout_rate: float
    
    # Spectral branch parameters
    n_spectral_features: int
    spectral_hidden_size: int
    spectral_norm_type: str  # 'batch', 'group', or 'none'
    spectral_dropout_rate: float
    
    # Fusion parameters
    concat_dropout_rate: float
    fusion_hidden_size: int
    fusion_norm_enabled: bool
    
    # Shared parameters
    activation: str
    gap_length: int
    
    @model_validator(mode="after")
    def validate_lengths(self):
        """Ensure all architecture lists have consistent lengths."""
        n_layers = len(self.n_filters)
        assert n_layers == 2, f"ShallowConcatterSE requires exactly 2 layers, got {n_layers}"
        assert len(self.kernel_sizes) == n_layers, "kernel_sizes length must match n_filters"
        assert len(self.strides) == n_layers, "strides length must match n_filters"
        assert len(self.paddings) == n_layers, "paddings length must match n_filters"
        return self
    
    @model_validator(mode="after")
    def validate_norm_types(self):
        """Validate normalization types."""
        assert self.raw_norm_type in ["batch", "group"], \
            f"raw_norm_type must be 'batch' or 'group', got {self.raw_norm_type}"
        assert self.spectral_norm_type in ["batch", "group", "none"], \
            f"spectral_norm_type must be 'batch', 'group', or 'none', got {self.spectral_norm_type}"
        return self
    
    @model_validator(mode="after")
    def validate_reduction_ratio(self):
        """Validate SE block reduction ratio."""
        if self.use_se_blocks:
            assert self.reduction_ratio > 0, "reduction_ratio must be positive when using SE blocks"
        return self


class ShallowConcatterSE(nn.Module):
    def __init__(self, cfg: ShallowConcatterSEConfig):
        super().__init__()
        
        # Get activation function
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
        
        # Build raw conv layers with optional SE blocks
        raw_conv_layers: List[nn.Module] = []
        
        # First conv layer
        raw_conv_layers.append(
            nn.Conv2d(
                in_channels=1,
                out_channels=cfg.n_filters[0],
                kernel_size=cfg.kernel_sizes[0],
                stride=cfg.strides[0],
                padding=cfg.paddings[0],
            )
        )
        
        # Normalization for first layer
        if cfg.raw_norm_type == "group":
            raw_conv_layers.append(
                nn.GroupNorm(num_groups=min(cfg.n_filters[0], 4), num_channels=cfg.n_filters[0])
            )
        elif cfg.raw_norm_type == "batch":
            raw_conv_layers.append(nn.BatchNorm2d(cfg.n_filters[0]))
        
        raw_conv_layers.append(act)
        
        # Optional SE block after first layer
        if cfg.use_se_blocks:
            raw_conv_layers.append(SEBlock(cfg.n_filters[0], cfg.reduction_ratio))
        
        raw_conv_layers.append(nn.Dropout(cfg.raw_dropout_rate))
        
        # Second conv layer
        raw_conv_layers.append(
            nn.Conv2d(
                in_channels=cfg.n_filters[0],
                out_channels=cfg.n_filters[1],
                kernel_size=cfg.kernel_sizes[1],
                stride=cfg.strides[1],
                padding=cfg.paddings[1],
            )
        )
        
        # Normalization for second layer
        if cfg.raw_norm_type == "group":
            raw_conv_layers.append(
                nn.GroupNorm(num_groups=min(cfg.n_filters[1], 4), num_channels=cfg.n_filters[1])
            )
        elif cfg.raw_norm_type == "batch":
            raw_conv_layers.append(nn.BatchNorm2d(cfg.n_filters[1]))
        
        raw_conv_layers.append(act)
        
        # Optional SE block after second layer
        if cfg.use_se_blocks:
            raw_conv_layers.append(SEBlock(cfg.n_filters[1], cfg.reduction_ratio))
        
        raw_conv_layers.append(nn.Dropout(cfg.raw_dropout_rate))
        
        self.raw_conv = nn.Sequential(*raw_conv_layers)
        
        # Global Average Pooling
        gap_size = (cfg.gap_length, cfg.gap_length)
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        
        # Build spectral network
        spec_net_layers: List[nn.Module] = [
            nn.Linear(cfg.n_spectral_features, cfg.spectral_hidden_size)
        ]
        
        if cfg.spectral_norm_type == "batch":
            spec_net_layers.append(nn.BatchNorm1d(cfg.spectral_hidden_size))
        elif cfg.spectral_norm_type == "group":
            spec_net_layers.append(
                nn.GroupNorm(
                    num_groups=min(cfg.spectral_hidden_size, 4),
                    num_channels=cfg.spectral_hidden_size
                )
            )
        elif cfg.spectral_norm_type == "none":
            pass  # No normalization
        
        spec_net_layers.extend([act, nn.Dropout(cfg.spectral_dropout_rate)])
        
        self.spec_net = nn.Sequential(*spec_net_layers)
        
        # Fusion head
        concatter_input_size = (cfg.n_filters[1] * gap_size[0] * gap_size[1]) + cfg.spectral_hidden_size
        
        fusion_layers: List[nn.Module] = [
            nn.Linear(concatter_input_size, cfg.fusion_hidden_size),
            act,
            nn.Dropout(cfg.concat_dropout_rate),
        ]
        
        if cfg.fusion_norm_enabled:
            fusion_layers.append(nn.BatchNorm1d(cfg.fusion_hidden_size))
        
        fusion_layers.append(nn.Linear(cfg.fusion_hidden_size, 1))
        
        self.fusion = nn.Sequential(*fusion_layers)
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        x: Tuple[torch.Tensor, torch.Tensor]
            x_raw: shape (batch, channels, time) or (batch, 1, channels, time)
            x_spectral: shape (batch, n_spectral_features)
        """
        x_raw, x_spectral = x
        
        # Ensure raw input has channel dimension
        if x_raw.dim() == 3:
            x_raw = x_raw.unsqueeze(1)
        
        # Process raw branch
        x_raw = self.raw_conv(x_raw)
        
        # MPS workaround: adaptive pooling requires divisible sizes on MPS
        if x_raw.device.type == 'mps':
            # logger.warning("MPS workaround: adaptive pooling requires divisible sizes on MPS")
            original_device = x_raw.device
            x_raw = self.gap(x_raw.cpu()).to(original_device)
        else:
            x_raw = self.gap(x_raw)
        
        x_raw = x_raw.flatten(1)
        
        # Process spectral branch
        x_spectral = self.spec_net(x_spectral)
        
        # Concatenate and fuse
        return self.fusion(torch.cat([x_raw, x_spectral], dim=1))

