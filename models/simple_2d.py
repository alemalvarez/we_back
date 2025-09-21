import torch
from torch import nn
from typing import List, Tuple

class Simple2D(nn.Module):
    """Simple 2D CNN for EEG data classification.
    This is thought for taking in data with a shape like
    (n_samples, n_channels, 1).

    We will convolve over the samples and channels with 
    2d covolutions. We kind of understand a EEg samples as a
    very long image, with a single channel.
    """

    def __init__(
        self,
        n_filters: List[int],
        kernel_sizes: List[Tuple[int, int]],
        strides: List[Tuple[int, int]],
        dropout_rate: float,
        input_shape: Tuple[int, int, int],
    ):
        super(Simple2D, self).__init__()

        self.input_shape = input_shape

        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = n_filters[0],
                kernel_size = kernel_sizes[0],
                stride = strides[0],
                padding = (0,0),
            ),
            nn.BatchNorm2d(n_filters[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_filters[0],
                out_channels = n_filters[1],
                kernel_size = kernel_sizes[1],
                stride = strides[1],
                padding = (0,0),
            ),
            nn.BatchNorm2d(n_filters[1]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_filters[1],
                out_channels = n_filters[2],
                kernel_size = kernel_sizes[2],
                stride = strides[2],
                padding = (0,0),
            ),
            nn.BatchNorm2d(n_filters[2]),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        self.gloval_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(n_filters[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is (batch, channels=1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.gloval_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

