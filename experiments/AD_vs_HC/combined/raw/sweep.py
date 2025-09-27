from datetime import datetime
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from core.raw_dataset import RawDataset
from models.simple_2d import Simple2D3Layers
from loguru import logger
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
import wandb
from dotenv import load_dotenv
import os
from core.run_sweep import run_sweep

load_dotenv()

WANDB_PROJECT = "AD_vs_HC_sweep"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")

def main():
    with wandb.init(project=WANDB_PROJECT) as run:
        config = run.config

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
        
        model = Simple2D3Layers(
            n_filters=config.n_filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            dropout_rate=config.dropout_rate
        )

        run_sweep(model, run, training_dataset, validation_dataset)
       
if __name__ == "__main__":
    main()