import os
from typing import List

from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    MultiDatasetConfig,
)
from core.logging import make_logger
from core.cv import run_cv

from models.shallow_concatter import ShallowerConcatterConfig

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

    train_subjects_path = os.getenv(
        "TRAIN_SUBJECTS",
        "experiments/AD_vs_HC/combined/multi/splits/training_subjects.txt",
    )
    val_subjects_path = os.getenv(
        "VAL_SUBJECTS",
        "experiments/AD_vs_HC/combined/multi/splits/validation_subjects.txt",
    )
    all_subjects = _read_subjects(train_subjects_path) + _read_subjects(val_subjects_path)

    n_folds = 5

    model_config = ShallowerConcatterConfig(
        model_name="ShallowerConcatter",
        n_filters=[16, 32],
        kernel_sizes=[(40, 2), (8, 5)],
        strides=[(12, 6), (10, 5)],
        raw_dropout_rate=0.39811932916850734,
        paddings=[(5, 0), (1, 1)],
        activation="leaky_relu",
        n_spectral_features=16,
        spectral_hidden_size=128,
        spectral_dropout_rate=0.2809098483832669,
        concat_dropout_rate=0.5642264679590964,
        fusion_hidden_size=256,
        gap_length=4,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=0.0036565664394494863,
        weight_decay=0.00017060568872516544,
        use_cosine_annealing=True,
        cosine_annealing_t_0=5,
        cosine_annealing_t_mult=1,
        cosine_annealing_eta_min=1e-6,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='fixed',
        pos_weight_value=1.0,
    )

    dataset_config = MultiDatasetConfig(
        h5_file_path=h5_file_path,
        raw_normalization='full',
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
        log_to_wandb=True,
        wandb_init=None,
    )

    magic_logger = make_logger(wandb_enabled=run_config.log_to_wandb, wandb_init=run_config.wandb_init)

    run_cv(
        all_subjects=all_subjects,
        n_folds=n_folds,
        run_config=run_config,
        magic_logger=magic_logger,   
    )

if __name__ == "__main__":
    main()