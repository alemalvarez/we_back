import torch
from torch import nn
from typing import List, Tuple, Literal

from core.schemas import NetworkConfig


class SimpleConcatterConfig(NetworkConfig):
    model_name: str = "SimpleConcatter"
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    raw_dropout_rate: float
    spectral_dropout_rate: float
    paddings: List[Tuple[int, int]]
    activation: str
    n_spectral_features: int
    head_hidden_sizes: List[int]
    concat_dropout_rate: float


class SimpleConcatter(nn.Module):
    def __init__(
        self,
        cfg: SimpleConcatterConfig,
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
            seq = nn.Sequential(conv, bn, act, nn.Dropout(cfg.raw_dropout_rate))
            layers.append(seq)
            in_channels = cfg.n_filters[i]

        self.layers = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Normalization before concatenation
        self.raw_norm = nn.BatchNorm1d(cfg.n_filters[3])
        self.spectral_norm = nn.BatchNorm1d(cfg.n_spectral_features)
        self.spectral_dropout = nn.Dropout(cfg.spectral_dropout_rate)

        # Build concatenation head with hidden layers
        head_layers: List[nn.Module] = []
        in_features = cfg.n_filters[3] + cfg.n_spectral_features
        for hidden_size in cfg.head_hidden_sizes:
            head_layers.append(nn.Linear(in_features, hidden_size))
            head_layers.append(act)
            head_layers.append(nn.Dropout(cfg.concat_dropout_rate))
            in_features = hidden_size
        head_layers.append(nn.Linear(in_features, 1))
        self.classifier = nn.Sequential(*head_layers)
        
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
        
        x_raw = self.layers(x_raw)
        x_raw = self.global_avg_pool(x_raw)
        x_raw = x_raw.flatten(1)
        x_raw = self.raw_norm(x_raw)

        x_spectral = self.spectral_dropout(x_spectral)
        x_spectral = self.spectral_norm(x_spectral)

        x_combined = torch.cat([x_raw, x_spectral], dim=1)
        return self.classifier(x_combined)

class ConcatterConfig(NetworkConfig):
    model_name: str = "Concatter"
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    raw_dropout_rate: float
    spectral_dropout_rate: float
    paddings: List[Tuple[int, int]]
    activation: str
    n_spectral_features: int
    head_hidden_sizes: List[int]
    concat_dropout_rate: float
    
    alpha: float # the balance between raw and spectral features (higher alpha means more raw importance)

class Concatter(nn.Module):
    """
    This network consists of two branches plus a concattenation head.
    Raw branch:
        - 4 conv layers in the style of deeper custom.
        - each layer: (conv, bn, act, dropout)
    Spectral branch:
        - Dropout layer for regularization.
    Concattenation head:
        - we pool the output of the cnns
        - we concatenate it to the spectral features based on a balance parameter alpha
        - multi-layer perceptron with dropout for output
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
            seq = nn.Sequential(conv, bn, act, nn.Dropout(cfg.raw_dropout_rate))
            layers.append(seq)
            in_channels = cfg.n_filters[i]

        self.layers = nn.ModuleList(layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_dropout = nn.Dropout(cfg.spectral_dropout_rate)

        self.alpha = torch.tensor(cfg.alpha) # not a param, sweepable

        # Build concatenation head with hidden layers
        head_layers: List[nn.Module] = []
        in_features = cfg.n_filters[3] + cfg.n_spectral_features
        for hidden_size in cfg.head_hidden_sizes:
            head_layers.append(nn.Linear(in_features, hidden_size))
            head_layers.append(act)
            head_layers.append(nn.Dropout(cfg.concat_dropout_rate))
            in_features = hidden_size
        head_layers.append(nn.Linear(in_features, 1))
        self.classifier = nn.Sequential(*head_layers)

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

        x_spectral = self.spectral_dropout(x_spectral)

        alpha = self.alpha.to(x_raw.device)
        x_combined = torch.cat([
            x_raw * alpha, 
            x_spectral * (1.0 - alpha.item())]
        , dim=1)
        return self.classifier(x_combined)

class GatedConcatterConfig(NetworkConfig):
    model_name: str = "GatedConcatter"
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    raw_dropout_rate: float
    spectral_dropout_rate: float
    paddings: List[Tuple[int, int]]
    activation: str
    n_spectral_features: int
    head_hidden_sizes: List[int]
    concat_dropout_rate: float
    gate_in_features: Literal["raw", "spectral", "both"]

class GatedConcatter(nn.Module):
    """
    This network consists of two branches plus a concattenation head.
    Raw branch:
        - 4 conv layers in the style of deeper custom.
        - each layer: (conv, bn, act, dropout)
    Spectral branch:
        - Dropout layer for regularization.
    Concattenation head:
        - we pool the output of the cnns
        - we determine alpha through a perceptron based on the features
        - we concatenate it to the spectral features based on the determined alpha
        - multi-layer perceptron with dropout for output
    """

    def __init__(
        self,
        cfg: GatedConcatterConfig,
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
            seq = nn.Sequential(conv, bn, act, nn.Dropout(cfg.raw_dropout_rate))
            layers.append(seq)
            in_channels = cfg.n_filters[i]

        self.layers = nn.ModuleList(layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_dropout = nn.Dropout(cfg.spectral_dropout_rate)

        self.gate_in_features = cfg.gate_in_features
        if self.gate_in_features == "raw":
            gate_in_features = cfg.n_filters[3] 
        elif self.gate_in_features == "spectral":
            gate_in_features = cfg.n_spectral_features
        elif self.gate_in_features == "both":
            gate_in_features = cfg.n_filters[3] + cfg.n_spectral_features
        else:
            raise ValueError(f"Unsupported gate_in_features: {self.gate_in_features}")
            
        self.alpha_perceptron = nn.Sequential(
            nn.Linear(gate_in_features, 1),
            nn.Sigmoid()
        )

        # Build concatenation head with hidden layers
        head_layers: List[nn.Module] = []
        in_features = cfg.n_filters[3] + cfg.n_spectral_features
        for hidden_size in cfg.head_hidden_sizes:
            head_layers.append(nn.Linear(in_features, hidden_size))
            head_layers.append(act)
            head_layers.append(nn.Dropout(cfg.concat_dropout_rate))
            in_features = hidden_size
        head_layers.append(nn.Linear(in_features, 1))
        self.classifier = nn.Sequential(*head_layers)

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

        x_spectral = self.spectral_dropout(x_spectral)

        if self.gate_in_features == "raw":
            gate_value = self.alpha_perceptron(x_raw)
        elif self.gate_in_features == "spectral":
            gate_value = self.alpha_perceptron(x_spectral)
        elif self.gate_in_features == "both":
            gate_value = self.alpha_perceptron(torch.cat([x_raw, x_spectral], dim=1))
        else:
            raise ValueError(f"Unsupported gate_in_features: {self.gate_in_features}")
        
        x_combined = torch.cat([
            x_raw * gate_value,
            x_spectral * (1 - gate_value)
        ], dim=1)
        return self.classifier(x_combined)

