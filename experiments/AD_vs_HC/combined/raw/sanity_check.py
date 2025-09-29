from dataclasses import dataclass
from typing import Tuple, Literal
from typing import List
from core.schemas import BaseModelConfig
from core.raw_dataset import RawDataset
from models.simple_2d import Deeper2D
from core.sanity_test_model import sanity_test_model
from core.validate_kernel import validate_kernel
import yaml # type: ignore[import]
import os
from dotenv import load_dotenv
from loguru import logger
load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")
WANDB_PROJECT = "AD_vs_HC"

# WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/simple2d3layers.yaml"), Loader=yaml.FullLoader)
# WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/improved2d.yaml"), Loader=yaml.FullLoader)
WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/deeper.yaml"), Loader=yaml.FullLoader)

WANDB_CONFIG["random_seed"] = int(os.getenv("RANDOM_SEED", "42"))

@dataclass
class Simple2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float

@dataclass
class Improved2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float
    paddings: List[Tuple[int, int]]

@dataclass
class Deeper2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float
    paddings: List[Tuple[int, int]]

# config = Simple2DConfig(**WANDB_CONFIG) # type: ignore
# config = Improved2DConfig(**WANDB_CONFIG) # type: ignore
config = Deeper2DConfig(**WANDB_CONFIG) # type: ignore

if not validate_kernel(config.kernel_sizes, config.strides, config.paddings, (1000, 68)):
    logger.error("Kernel configuration is invalid")
    exit(1)

logger.success("Kernel configuration is valid")

# model = Simple2D3Layers(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate)
# model = Improved2D(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate, paddings=config.paddings)
model = Deeper2D(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate, paddings=config.paddings)


dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
    normalize=config.normalize
)

sanity_test_model(
    model, 
    config, 
    dataset, 
    run_overfit_test=True, 
    overfit_test_epochs=100
)