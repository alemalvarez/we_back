import torch
from torch import nn
from typing import List, Tuple, Optional

class DeeperCustomConcat(nn.Module):
    """
    2D CNN for EEG data classification with support for concatenating
    additional spectral features before the final classifier.
    Expects input as a tuple: (raw_tensor, spectral_features)
    - raw_tensor: shape (batch, channels, time) or (batch, 1, channels, time)
    - spectral_features: shape (batch, n_spectral_features)
    """

    def __init__(
        self,
        n_filters: List[int],
        kernel_sizes: List[Tuple[int, int]],
        strides: List[Tuple[int, int]],
        dropout_rate: float,
        paddings: List[Tuple[int, int]],
        activation: str,
        n_spectral_features: int,
    ):
        super().__init__()

        # Select activation
        if activation == "relu":
            act: nn.Module = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act = nn.LeakyReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.activation = act

        # Build convolutional layers
        layers = []
        in_channels = 1
        for i in range(4):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=n_filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
            )
            bn = nn.BatchNorm2d(n_filters[i])
            seq = nn.Sequential(conv, bn, nn.Dropout(dropout_rate), act)
            layers.append(seq)
            in_channels = n_filters[i]

        self.layers = nn.ModuleList(layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # The input to the classifier is conv_out + n_spectral_features
        self.classifier = nn.Linear(n_filters[3] + n_spectral_features, 1)

    def forward(
        self,
        x: torch.Tensor,
        spectral_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: raw input, shape (batch, channels, time) or (batch, 1, channels, time)
        spectral_features: shape (batch, n_spectral_features)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        if spectral_features is not None:
            x = torch.cat([x, spectral_features], dim=1)
        return self.classifier(x)
