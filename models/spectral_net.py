import torch
from torch import nn
from typing import List
from core.schemas import NetworkConfig


class SpectralNetConfig(NetworkConfig):
    model_name: str = "SpectralNet"
    input_size: int = 16
    hidden_1_size: int = 32
    hidden_2_size: int = 16
    dropout_rate: float = 0.5

class SpectralNet(nn.Module):
    def __init__(self,
        config: SpectralNetConfig,
    ):
        super(SpectralNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_1_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(config.hidden_1_size, config.hidden_2_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),

            nn.Linear(config.hidden_2_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x


class AdvancedSpectralNetConfig(NetworkConfig):
    model_name: str = "AdvancedSpectralNet"
    input_size: int = 16
    hidden_1_size: int = 16
    hidden_2_size: int = 32
    dropout_rate: float = 0.5
    add_batch_norm: bool = False
    activation: str = "relu"

class AdvancedSpectralNet(nn.Module):
    def __init__(self,
        config: AdvancedSpectralNetConfig,
    ):
        super(AdvancedSpectralNet, self).__init__()
        
        if config.activation == "relu":
            act: nn.Module = nn.ReLU(inplace=True)
        elif config.activation == "leaky_relu":
            act = nn.LeakyReLU(inplace=True)
        elif config.activation == "gelu":
            act = nn.GELU()
        elif config.activation == "silu":
            act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

        layers: List[nn.Module] = []
        
        layers.append(nn.Linear(config.input_size, config.hidden_1_size))
        if config.add_batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_1_size))
        layers.append(act)
        layers.append(nn.Dropout(config.dropout_rate))
        layers.append(nn.Linear(config.hidden_1_size, config.hidden_2_size))
        if config.add_batch_norm:
            layers.append(nn.BatchNorm1d(config.hidden_2_size))
        layers.append(act)
        layers.append(nn.Dropout(config.dropout_rate))
        layers.append(nn.Linear(config.hidden_2_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x