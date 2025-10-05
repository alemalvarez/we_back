import torch
from torch import nn

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