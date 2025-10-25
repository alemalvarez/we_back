import os
from typing import Literal, List

import wandb

from core.cv import run_cv
from core.logging import make_logger
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    MultiDatasetConfig,
)
from core.validate_kernel import validate_kernel
from models.shallow_concatter import ShallowerConcatterConfig
from dotenv import load_dotenv

load_dotenv()

def _parse_two_level(s: str) -> list:
    """Parse two-level string parameters from wandb sweeps."""
    s = s.strip()
    if "__" in s:
        return [[int(p) for p in g.split("_") if p] for g in s.split("__") if g]
    return [int(p) for p in s.split("_") if p]

def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    n_filters = _parse_two_level(cfg.get("n_filters", "16_32_64_128"))
    kernels = _parse_two_level(cfg.get("kernel_sizes", "100_3__15_10__10_3__5_2"))
    strides = _parse_two_level(cfg.get("strides", "2_2__2_2__1_1__1_1"))
    paddings = _parse_two_level(cfg.get("paddings", "25_1__5_2__5_1__1_1"))

    input_shape = (1000, 68)

    validate_kernel(kernels, strides, paddings, input_shape)

    network_config = ShallowerConcatterConfig(
        model_name="ShallowerConcatter",
        n_filters=[16,32],
        kernel_sizes=[(40, 2), (8, 5)],
        strides=[(12, 6), (10, 5)],
        raw_dropout_rate=float(cfg.get("raw_dropout_rate", 0.25)),
        paddings=[(5, 0), (1, 1)],
        activation=cfg.get("activation", "silu"),
        n_spectral_features=16,
        spectral_hidden_size=int(cfg.get("spectral_hidden_size", 32)),
        concat_dropout_rate=float(cfg.get("concat_dropout_rate", 0.25)),
        spectral_dropout_rate=float(cfg.get("spectral_dropout_rate", 0.25)),
        fusion_hidden_size=int(cfg.get("fusion_hidden_size", 128)),
        gap_length=4,
        raw_norm_type=cfg.get("raw_norm_type", "group"),
        spectral_norm_type=cfg.get("spectral_norm_type", "none"),
        fusion_norm_enabled=bool(cfg.get("fusion_norm_enabled", False)),
    )

    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate", 3e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        use_cosine_annealing=False,
    )

    criterion_config = CriterionConfig(
        pos_weight_type="multiplied",
        pos_weight_value=1.0
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
        random_seed=int(os.getenv("RANDOM_SEED", 42)),
        batch_size=int(cfg.get("batch_size", 32)),
        max_epochs=75, # here i just saw a run hit this limit. it should also go up.
        patience=8, # im afraid this is too low. with shorter runs (smaller models), we might be able to afford at least 10.
        min_delta=0.001,
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