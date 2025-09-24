from dataclasses import dataclass
from typing import Tuple, Literal
from typing import List
from core.schemas import BaseModelConfig
from core.raw_dataset import RawDataset
from models.simple_2d import Simple2D
from core.sanity_test_model import sanity_test_model

@dataclass
class Simple2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    input_shape: Tuple[int, int, int]
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']

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

config = Simple2DConfig(**WANDB_CONFIG) # type: ignore

model = Simple2D(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate, input_shape=config.input_shape)

dataset = RawDataset(
    h5_file_path="h5test_raw_only.h5",
    subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/raw/splits/training_subjects.txt",
    normalize=config.normalize
)

sanity_test_model(
    model, 
    config, 
    dataset, 
    run_overfit_test=False, 
    overfit_test_epochs=100
)