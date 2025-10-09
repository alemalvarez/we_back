import os
from typing import List

from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    RawDatasetConfig,
)
from core.logging import make_logger
from core.cv import run_cv

from models.simple_2d import DeeperCustomConfig

from dotenv import load_dotenv
load_dotenv()

def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "h5test_raw_only.h5",
    )

    splits_dir = "experiments/AD_vs_HC/combined/raw/splits"
    train_subjects_path = os.path.join(splits_dir, "training_subjects.txt")
    val_subjects_path = os.path.join(splits_dir, "validation_subjects.txt")
    all_subjects = _read_subjects(train_subjects_path) + _read_subjects(val_subjects_path)

    n_folds = 5

    model_config = DeeperCustomConfig(
        model_name="DeeperCustom",
        n_filters=[16, 32, 64, 128],
        kernel_sizes=[(100, 3), (15, 10), (10, 3), (5, 2)],
        strides=[(2, 2), (2, 2), (1, 1), (1, 1)],
        dropout_rate=0.31158910319253397,
        paddings=[(25, 1), (5, 2), (5, 1), (1, 1)],
        activation="silu",
        dropout_before_activation=False,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=0.004255107493153422,
        weight_decay=9.6832252733516e-05,
        use_cosine_annealing=True,
        cosine_annealing_t_0=8,
        cosine_annealing_t_mult=2,
        cosine_annealing_eta_min=1e-6,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='multiplied',
        pos_weight_value=1.09784373282656,
    )

    dataset_config = RawDatasetConfig(
        h5_file_path=h5_file_path,
        raw_normalization='channel-subject',
        augment=False,
        augment_prob=(0.5, 0.5),
        noise_std=0.1,
    )

    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=42,
        batch_size=128,
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