from core.raw_dataset import RawDataset
from core.model_playground import load_config, create_model
from core.train_model import train_model
from dotenv import load_dotenv
import os
import sys
from loguru import logger
import torch
import numpy as np

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")

# Get config file path from command line argument
if len(sys.argv) != 2:
    logger.error("Usage: python train.py <config_file>")
    logger.error("Example: python train.py configs/Improved2D_0928_2024.yaml")
    sys.exit(1)

config_file = sys.argv[1]
logger.info(f"Using config file: {config_file}")

# Load configuration using the universal loader
config = load_config(config_file)

torch.manual_seed(int(os.getenv("RANDOM_SEED", "42")))
np.random.seed(int(os.getenv("RANDOM_SEED", "42")))



# Create model using the universal creator
model = create_model(config)

training_dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
    normalize=getattr(config, "normalize", "sample-channel"),  # type: ignore[attr-defined]
    augment=bool(getattr(config, "augment", False)),  # type: ignore[attr-defined]
    augment_prob=(
        float(getattr(config, "augment_prob_neg", 0.5)),  # type: ignore[attr-defined] # neg, pos
        float(getattr(config, "augment_prob_pos", 0.0))  # type: ignore[attr-defined]
    ),
    noise_std=float(getattr(config, "noise_std", 0.1))  # type: ignore[attr-defined]
)

validation_dataset = RawDataset(
    h5_file_path=H5_FILE_PATH,
    subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
    normalize=getattr(config, "normalize", "sample-channel"),  # type: ignore[attr-defined]
    augment=False
)

trained_model = train_model(model, config, training_dataset, validation_dataset)

# Save the trained model next to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, f"{config.model_name}_trained.pt")
torch.save(trained_model.state_dict(), save_path)
logger.success(f"Saved trained model to {save_path}")