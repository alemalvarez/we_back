import os
from typing import List, Literal

import wandb

from core.logging import make_logger
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    MultiDatasetConfig,
)
from core.cv import run_cv
from models.concatter import ConcatterConfig
from dotenv import load_dotenv

load_dotenv()


def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    network_config = ConcatterConfig(
        model_name="Concatter",
        n_filters=[16, 32, 64, 128],
        kernel_sizes=[(100, 3), (15, 10), (10, 3), (5, 2)],
        strides=[(2, 2), (2, 2), (1, 1), (1, 1)],
        dropout_rate=float(cfg.get("dropout_rate", 0.25)),
        paddings=[(25, 1), (5, 2), (5, 1), (1, 1)],
        activation="silu",
        n_spectral_features=16,
        spectral_dropout_rate=0.5,
        head_hidden_sizes=[128, 32],
        raw_weight=0.6,
        spectral_weight=0.4,
    )

    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate", 3e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)) if cfg.get("weight_decay") is not None else None,
        use_cosine_annealing=bool(cfg.get("use_cosine_annealing", False)),
        cosine_annealing_t_0=int(cfg.get("cosine_annealing_t_0", 5)),
        cosine_annealing_t_mult=int(cfg.get("cosine_annealing_t_mult", 1)),
        cosine_annealing_eta_min=float(cfg.get("cosine_annealing_eta_min", 1e-6)),
    )

    criterion_config = CriterionConfig(
        pos_weight_type="multiplied",
        pos_weight_value=float(cfg.get("pos_weight_value", 1.0)),
    )

    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5",
    )

    raw_norm_lit: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full'] = cfg.get("raw_normalization", "channel-subject")  # type: ignore

    run_config = RunConfig(
        network_config=network_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=int(cfg.get("random_seed", 42)),
        batch_size=int(cfg.get("batch_size", 32)),
        max_epochs=int(cfg.get("max_epochs", 50)),
        patience=int(cfg.get("patience", 15)),
        min_delta=float(cfg.get("min_delta", 0.001)),
        early_stopping_metric="loss",
        dataset_config=MultiDatasetConfig(
            h5_file_path=h5_file_path,
            spectral_normalization="standard",
            raw_normalization=raw_norm_lit,
        ),
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
        "experiments/AD_vs_HC/combined/multi/splits/training_subjects.txt",
    )
    val_subjects_path = os.getenv(
        "VAL_SUBJECTS",
        "experiments/AD_vs_HC/combined/multi/splits/validation_subjects.txt",
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