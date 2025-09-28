import torch
from torch import nn
from typing import List, Tuple


class Deeper2D(nn.Module):
    """Simple 2D CNN for EEG data classification.
    This is thought for taking in data with a shape like
    (n_samples, n_channels, 1).
    """
    
    def __init__(
        self,
        n_filters: List[int],
        kernel_sizes: List[Tuple[int, int]],
        strides: List[Tuple[int, int]],
        dropout_rate: float,
        paddings: List[Tuple[int, int]],
    ):
        super(Deeper2D, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = n_filters[0],
                kernel_size = kernel_sizes[0],
                stride = strides[0],
                padding = paddings[0],
            ),
            nn.BatchNorm2d(n_filters[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_filters[0],
                out_channels = n_filters[1],
                kernel_size = kernel_sizes[1],
                stride = strides[1],
                padding = paddings[1],
            ),
            nn.BatchNorm2d(n_filters[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_filters[1],
                out_channels = n_filters[2],
                kernel_size = kernel_sizes[2],
                stride = strides[2],
                padding = paddings[2],
            ),
            nn.BatchNorm2d(n_filters[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_filters[2],
                out_channels = n_filters[3],
                kernel_size = kernel_sizes[3],
                stride = strides[3],
                padding = paddings[3],
            ),
            nn.BatchNorm2d(n_filters[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(n_filters[3], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

class Improved2D(nn.Module):
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
        paddings: List[Tuple[int, int]],
    ):
        super(Improved2D, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = n_filters[0],
                kernel_size = kernel_sizes[0],
                stride = strides[0],
                padding = paddings[0],
            ),
            nn.BatchNorm2d(n_filters[0]),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_filters[0],
                out_channels = n_filters[1],
                kernel_size = kernel_sizes[1],
                stride = strides[1],
                padding = paddings[1],
            ),
            nn.BatchNorm2d(n_filters[1]),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(
                in_channels = n_filters[1],
                out_channels = n_filters[2],
                kernel_size = kernel_sizes[2],
                stride = strides[2],
                padding = paddings[2],
            ),
            nn.BatchNorm2d(n_filters[2]),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(n_filters[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is (batch, channels=1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.l3(self.l2(self.l1(x)))
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)



class Simple2D3Layers(nn.Module):
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
    ):
        super(Simple2D3Layers, self).__init__()

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
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(n_filters[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is (batch, channels=1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.l3(self.l2(self.l1(x)))
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

