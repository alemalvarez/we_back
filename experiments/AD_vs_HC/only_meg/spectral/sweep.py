import os
from typing import List, Literal

import wandb

from core.logging import make_logger
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    SpectralDatasetConfig,
)
from core.cv import run_cv
from models.spectral_net import AdvancedSpectralNetConfig
from dotenv import load_dotenv

load_dotenv()


def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    network_config = AdvancedSpectralNetConfig(
        model_name="AdvancedSpectralNet",
        input_size=16,
        hidden_1_size=int(cfg.get("hidden_1_size", 16)),
        hidden_2_size=int(cfg.get("hidden_2_size", 32)),
        dropout_rate=float(cfg.get("dropout_rate", 0.5)),
        add_batch_norm=True,
        activation=cfg.get("activation", "relu"),
    )

    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate", 3e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)) if cfg.get("weight_decay") is not None else None,
        use_cosine_annealing=False,
    )

    criterion_config = CriterionConfig(
        pos_weight_type="multiplied",
        pos_weight_value=1.0,
    )

    norm_str = str(cfg.get("normalization", "standard"))
    norm_lit: Literal['min-max','standard','none'] = norm_str if norm_str in ("min-max","standard","none") else "standard"  # type: ignore[assignment]

    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5",
    )

    h5_file_path = "harmonized_raw_features.h5"

    run_config = RunConfig(
        network_config=network_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=int(os.getenv("RANDOM_SEED", 42)),
        batch_size=128,
        max_epochs=500,
        patience=15,
        min_delta=0.001,
        early_stopping_metric="loss",
        dataset_config=SpectralDatasetConfig(
            h5_file_path=h5_file_path,
            spectral_normalization=norm_lit,
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
        "experiments/AD_vs_HC/combined/spectral/splits/training_subjects.txt",
    )
    val_subjects_path = os.getenv(
        "VAL_SUBJECTS",
        "experiments/AD_vs_HC/combined/spectral/splits/validation_subjects.txt",
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
        min_fold_mcc=.40,
    )

    magic_logger.finish()


if __name__ == "__main__":
    main()