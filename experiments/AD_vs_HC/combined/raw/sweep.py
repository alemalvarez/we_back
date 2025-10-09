import os
from typing import List

import wandb

from core.logging import make_logger
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    RawDatasetConfig,
)
from core.cv import run_cv
from models.simple_2d import DeeperCustomConfig
from dotenv import load_dotenv

load_dotenv()


def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    network_config = DeeperCustomConfig(
        model_name="DeeperCustom",
        n_filters=[16, 32, 64, 128],
        kernel_sizes=[(100, 3), (15, 10), (10, 3), (5, 2)],
        strides=[(2, 2), (2, 2), (1, 1), (1, 1)],
        paddings=[(25, 1), (5, 2), (5, 1), (1, 1)],
        activation=cfg.get("activation"), # part of the sweep
        dropout_before_activation=False,
        dropout_rate=float(cfg.get("dropout_rate")), # part of the sweep
    )

    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate")), # part of the sweep
        weight_decay=0.00005,
        use_cosine_annealing=True,
        cosine_annealing_t_0=7,
        cosine_annealing_t_mult=2,
        cosine_annealing_eta_min=1e-6,
    )

    criterion_config = CriterionConfig(
        pos_weight_type="multiplied",
        pos_weight_value=1.0,
    )

    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5",
    )

    dataset_config = RawDatasetConfig(
        h5_file_path=h5_file_path,
        raw_normalization="channel-subject", 
        augment=True,
        augment_prob=(cfg.get("augment_prob_pos"), cfg.get("augment_prob_neg")), # part of the sweep
        noise_std=cfg.get("noise_std"), # part of the sweep
    )

    run_config = RunConfig(
        network_config=network_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=int(os.getenv("RANDOM_SEED", 42)),
        batch_size=int(cfg.get("batch_size", 32)), # part of the sweep
        max_epochs=50,
        patience=5,
        min_delta=.001,
        early_stopping_metric="loss",
        dataset_config=dataset_config,
        log_to_wandb=True,
        wandb_init=None,
    )
    return run_config


def main() -> None:
    # Expect a W&B agent to have initialized the run; otherwise, init minimally
    if wandb.run is None:
        wandb.init(project=os.getenv("WANDB_PROJECT", "AD_vs_HC"))

    cfg = wandb.config

    train_subjects_path = os.getenv(
        "TRAIN_SUBJECTS",
        "experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
    )
    val_subjects_path = os.getenv(
        "VAL_SUBJECTS",
        "experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
    )

    all_subjects = _read_subjects(train_subjects_path) + _read_subjects(val_subjects_path)
    n_folds = 5

    run_config = build_run_config_from_wandb(cfg)
    
    magic_logger = make_logger(wandb_enabled=run_config.log_to_wandb, wandb_init=run_config.wandb_init)

    run_cv(
        all_subjects=all_subjects,
        n_folds=n_folds,
        run_config=run_config,
        magic_logger=magic_logger,
        min_fold_mcc=.37,
    )

    magic_logger.finish()


if __name__ == "__main__":
    main()