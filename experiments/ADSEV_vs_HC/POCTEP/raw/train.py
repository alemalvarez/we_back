from typing import Tuple, List, Literal
from dataclasses import dataclass

from core.raw_dataset import RawDataset
from core.schemas import BaseModelConfig
from models.simple_2d import Simple2D
from core.train_model import train_model

H5_FILE_PATH = "h5test_raw_only.h5"
WANDB_PROJECT = "ADSEV_vs_HC"
WANDB_CONFIG = {
    "random_seed": 42,
    "model_name": "Simple2D_3layers",
    "n_filters": [16, 32, 64],
    "kernel_sizes": [(5, 5), (5, 5), (5, 5)],
    "strides": [(1, 1), (1, 1), (1, 1)],
    "dropout_rate": 0.25,
    "input_shape": (1000, 1, 68),
    "learning_rate": 0.001,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 20,
    "min_delta": 0.001,
    "normalize": "sample-channel",
}

@dataclass
class Simple2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    input_shape: Tuple[int, int, int]
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']


config = Simple2DConfig(**WANDB_CONFIG) # type: ignore
model = Simple2D(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate, input_shape=config.input_shape)

training_dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/raw/splits/training_subjects.txt",
    normalize=config.normalize
)

validation_dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/raw/splits/validation_subjects.txt",
    normalize=config.normalize
)

train_model(model, config, training_dataset, validation_dataset)