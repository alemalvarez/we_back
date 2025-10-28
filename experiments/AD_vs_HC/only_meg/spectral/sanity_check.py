from dataclasses import dataclass

from core.spectral_dataset import SpectralDataset
from models.spectral_net import SpectralNet
from core.schemas import BaseModelConfig
import os
from dotenv import load_dotenv
from core.sanity_test_model import sanity_test_model

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")

# Configuration constants
WANDB_PROJECT = "AD_vs_HC"
WANDB_CONFIG = {
    "random_seed": 42,
    "model_name": "SpectralNet_2layers",
    "input_size": 16,
    "hidden_1_size": 32,
    "hidden_2_size": 16,
    "dropout_rate": 0.0031824642710681,
    "learning_rate": 0.02111076215981144,
    "weight_decay": 0.0000027819671966625116,
    "batch_size": 64,
    "max_epochs": 50,
    "patience": 5,
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
model = SpectralNet(
    input_size=config.input_size,
    hidden_1_size=config.hidden_1_size,
    hidden_2_size=config.hidden_2_size,
    dropout_rate=config.dropout_rate
)

sanity_test_model(model, config, training_dataset, run_overfit_test=True, overfit_test_epochs=30)