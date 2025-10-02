from typing import List, Tuple, Literal
from dataclasses import dataclass
from core.schemas import BaseModelConfig
from dotenv import load_dotenv
import os
import yaml # type: ignore[import]
from core.run_cv import run_cv
from models.simple_2d import DeeperCustom
from loguru import logger
load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")
WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/custom.yaml"), Loader=yaml.FullLoader)
N_FOLDS = 5

WANDB_CONFIG["random_seed"] = int(os.getenv("RANDOM_SEED", "42"))

@dataclass
class CustomConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float
    paddings: List[Tuple[int, int]]
    activation: str
    augment: bool
    augment_prob_pos: float
    augment_prob_neg: float
    noise_std: float

config = CustomConfig(**WANDB_CONFIG) # type: ignore

model = DeeperCustom(n_filters=config.n_filters, kernel_sizes=config.kernel_sizes, strides=config.strides, dropout_rate=config.dropout_rate, paddings=config.paddings, activation=config.activation)
included_subjects: List[str] = []

for path in [
    "experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
    "experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
]:
    with open(path, "r") as f:
        included_subjects.extend([line.strip() for line in f if line.strip()])

run_cv(
    model=model,
    config=config,
    h5_file_path=H5_FILE_PATH,
    included_subjects=included_subjects,
    n_folds=N_FOLDS
)