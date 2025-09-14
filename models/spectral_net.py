import torch
from torch import nn


class SpectralNet(nn.Module):
    def __init__(self,
        input_size: int = 16,
        hidden_1_size: int = 32,
        hidden_2_size: int = 16,
        dropout_rate: float = 0.5,
    ):
        super(SpectralNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_1_size, hidden_2_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_2_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x