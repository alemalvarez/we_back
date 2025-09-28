from typing import Tuple, List, Literal
from dataclasses import dataclass

from core.raw_dataset import RawDataset
from core.schemas import BaseModelConfig
from models.simple_2d import Improved2D, Deeper2D
from core.train_model import train_model
from dotenv import load_dotenv
import os
import yaml # type: ignore[import]
from loguru import logger
import torch

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")
WANDB_PROJECT = "AD_vs_HC"
# WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/simple2d3layers.yaml"), Loader=yaml.FullLoader)
# WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/improved2d.yaml"), Loader=yaml.FullLoader)
WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/deeper.yaml"), Loader=yaml.FullLoader)
WANDB_CONFIG["random_seed"] = int(os.getenv("RANDOM_SEED", "42"))

@dataclass
class Simple2D3LayersConfig(BaseModelConfig):
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

# config = Simple2D3LayersConfig(**WANDB_CONFIG) # type: ignore
# model = Simple2D3Layers(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate)

# config = Improved2DConfig(**WANDB_CONFIG) # type: ignore
config = Deeper2DConfig(**WANDB_CONFIG) # type: ignore
model = Deeper2D(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate, paddings=config.paddings)

training_dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
    normalize=config.normalize
)

validation_dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
    normalize=config.normalize
)

trained_model = train_model(model, config, training_dataset, validation_dataset)

# Save the trained model next to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, f"{config.model_name}_trained.pt")
torch.save(trained_model.state_dict(), save_path)
logger.success(f"Saved trained model to {save_path}")