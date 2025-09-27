import torch
import numpy as np
from loguru import logger
from torch.utils.data import Subset
from dataclasses import dataclass

from core.spectral_dataset import SpectralDataset
from models.spectral_net import SpectralNet
from core.schemas import BaseModelConfig
import os
from dotenv import load_dotenv
from core.train_model import train_model

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")

# Configuration constants
WANDB_PROJECT = "AD_vs_HC"
WANDB_CONFIG = {
    "random_seed": 42,
    "model_name": "SpectralNet_2layers",
    "input_size": 16,
    "hidden_1_size": 16,
    "hidden_2_size": 16,
    "dropout_rate": 0.241824642710681,
    "learning_rate": 0.003111076215981144,
    "weight_decay": 0.00027819671966625116,
    "batch_size": 32,
    "max_epochs": 50,
    "patience": 15,
    "min_delta": 0.001,
}

@dataclass
class SpectralNetConfig(BaseModelConfig):
    input_size: int
    hidden_1_size: int
    hidden_2_size: int
    dropout_rate: float
    weight_decay: float

config = SpectralNetConfig(**WANDB_CONFIG) # type: ignore

training_dataset = SpectralDataset(
    h5_file_path=H5_FILE_PATH, 
    subjects_txt_path="experiments/AD_vs_HC/combined/spectral/splits/training_subjects.txt",
    normalize="standard"
)

validation_dataset = SpectralDataset(
    h5_file_path=H5_FILE_PATH, 
    subjects_txt_path="experiments/AD_vs_HC/combined/spectral/splits/validation_subjects.txt",
    normalize="standard"
)

# pos_indices = np.where(training_dataset.labels == 1)[0]
# neg_indices = np.where(training_dataset.labels == 0)[0]

# logger.info(f"Positives: {len(pos_indices)}, Negatives: {len(neg_indices)}")

# n_neg = len(neg_indices)
# pos_subset = np.random.choice(pos_indices, size=n_neg, replace=False)

# balanced_indices = np.concatenate([pos_subset, neg_indices])

# balanced_training_dataset = Subset(training_dataset, balanced_indices)

model = SpectralNet(
    input_size=config.input_size,
    hidden_1_size=config.hidden_1_size,
    hidden_2_size=config.hidden_2_size,
    dropout_rate=config.dropout_rate
)

trained_model = train_model(model, config, training_dataset, validation_dataset)

# Save the trained model next to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, f"{config.model_name}_trained.pt")
torch.save(trained_model.state_dict(), save_path)