import os
from typing import Tuple, List, Literal

import numpy as np
import torch
from dataclasses import dataclass
from loguru import logger
from dotenv import load_dotenv
import yaml  # type: ignore[import]

from core.eval_model import evaluate_model
from core.raw_dataset import RawDataset
from core.schemas import BaseModelConfig
from models.simple_2d import Deeper2D


load_dotenv()


@dataclass
class Deeper2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float
    paddings: List[Tuple[int, int]]


# Choose your checkpoint path here (no CLI args)
CKPT_PATH = "checkpoints/model-0.pt"

# Config used to instantiate the model (matches training)
WANDB_CONFIG = yaml.load(open("experiments/AD_vs_HC/combined/raw/deeper.yaml"), Loader=yaml.FullLoader)
WANDB_CONFIG["random_seed"] = int(os.getenv("RANDOM_SEED", "42"))

torch.manual_seed(WANDB_CONFIG["random_seed"])  # type: ignore[index]
np.random.seed(WANDB_CONFIG["random_seed"])  # type: ignore[index]

config = Deeper2DConfig(**WANDB_CONFIG)  # type: ignore[arg-type]

model = Deeper2D(
    n_filters=config.n_filters,
    kernel_sizes=config.kernel_sizes,
    strides=config.strides,
    dropout_rate=config.dropout_rate,
    paddings=config.paddings,
)

assert os.path.exists(CKPT_PATH), f"Checkpoint not found: {CKPT_PATH}" # type: ignore
state = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state)
logger.success(f"Loaded weights from {CKPT_PATH}")

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
validation_dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
    normalize=config.normalize,
)

evaluate_model(model, validation_dataset, batch_size=config.batch_size)


