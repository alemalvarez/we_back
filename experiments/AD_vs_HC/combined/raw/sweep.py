from core.raw_dataset import RawDataset
from core.model_playground import create_model_from_wandb_config
from loguru import logger
import wandb
from dotenv import load_dotenv
import os
from core.run_sweep import run_sweep
import torch
import numpy as np
import sys

load_dotenv()

WANDB_PROJECT = "AD_vs_HC_sweep"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")

def _parse_two_level(s: str) -> list:
    s = s.strip()
    if "__" in s:
        return [[int(p) for p in g.split("_") if p] for g in s.split("__") if g]
    return [int(p) for p in s.split("_") if p]

def main():
    # Get model type from command line argument if provided
    model_type = None
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        logger.info(f"Using model type: {model_type}")

    with wandb.init(project=WANDB_PROJECT) as run:
        config = run.config

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # Create training dataset with optional augment parameters
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

        # Create model using the universal creator
        model = create_model_from_wandb_config(config, model_type)

        run_sweep(model, run, training_dataset, validation_dataset)

if __name__ == "__main__":
    main()