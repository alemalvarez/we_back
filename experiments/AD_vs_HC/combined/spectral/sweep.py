import wandb
import os
from dotenv import load_dotenv
from loguru import logger

from core.spectral_dataset import SpectralDataset
from models.spectral_net import SpectralNet
from core.run_sweep import run_sweep

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
WANDB_PROJECT = "AD_vs_HC_sweep"
logger.info(f"H5 file path: {H5_FILE_PATH}")

def main():

    with wandb.init(project=WANDB_PROJECT) as run:
        config = run.config

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

        model = SpectralNet(
            input_size=16,
            hidden_1_size=config.hidden_1_size,
            hidden_2_size=config.hidden_2_size,
            dropout_rate=config.dropout_rate
        )

        run_sweep(model, run, training_dataset, validation_dataset)


if __name__ == "__main__":
    main()