import os
from typing import List

from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    SpectralDatasetConfig,
)
from core.logging import make_logger
from core.cv import run_cv

from models.spectral_net import SpectralNetConfig

from dotenv import load_dotenv
load_dotenv()

def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5",
    )

    splits_dir = "experiments/AD_vs_HC/combined/spectral/splits"
    train_subjects_path = os.path.join(splits_dir, "training_subjects.txt")
    val_subjects_path = os.path.join(splits_dir, "validation_subjects.txt")
    all_subjects = _read_subjects(train_subjects_path) + _read_subjects(val_subjects_path)

    n_folds = 5

    model_config = SpectralNetConfig()
    optimizer_config = OptimizerConfig(
        learning_rate=3e-2,
        weight_decay=None,
        use_cosine_annealing=True,
        cosine_annealing_t_0=5,
        cosine_annealing_t_mult=1,
        cosine_annealing_eta_min=1e-6,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='fixed',
        pos_weight_value=1.0,
    )

    dataset_config = SpectralDatasetConfig(
        h5_file_path=h5_file_path,
        spectral_normalization='standard',
    )

    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=42,
        batch_size=32,
        max_epochs=50,
        patience=5,
        min_delta=0.001,
        early_stopping_metric='loss',
        dataset_config=dataset_config,
        log_to_wandb=False,
        wandb_init=None,
    )

    magic_logger = make_logger(wandb_enabled=False, wandb_init=None)

    run_cv(
        all_subjects=all_subjects,
        n_folds=n_folds,
        run_config=run_config,
        magic_logger=magic_logger,   
    )

if __name__ == "__main__":
    main()