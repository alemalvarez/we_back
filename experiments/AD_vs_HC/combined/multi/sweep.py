import os
from typing import Literal, List

import wandb

from core.logging import make_logger
from core.schemas import (
    NetworkConfig,
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    MultiDatasetConfig,
)
from core.builders import build_dataset
from models.concatter import ConcatterConfig, GatedConcatterConfig
from core.runner import run as run_single
from core.evaluation import evaluate_with_config, pretty_print_per_subject
from dotenv import load_dotenv

load_dotenv()


def _build_head_hidden_sizes_from_str(head_hidden_sizes: str) -> List[int]:
    return [int(size) for size in head_hidden_sizes.split("_")]

def _build_network_from_wandb(cfg: wandb.Config) -> NetworkConfig:  # type: ignore[name-defined]
    arch_string = cfg.get("arch")
    if arch_string.startswith("concatter"):
        alpha = arch_string.split("_")[1]
        return ConcatterConfig(
            model_name="Concatter",
            n_filters=[16, 32, 64, 128],
            kernel_sizes=[(100, 3), (15, 10), (10, 3), (5, 2)],
            strides=[(2, 2), (2, 2), (1, 1), (1, 1)],
            raw_dropout_rate=float(cfg.get("raw_dropout_rate", 0.25)),
            paddings=[(25, 1), (5, 2), (5, 1), (1, 1)],
            activation="silu",
            n_spectral_features=16,
            spectral_dropout_rate=float(cfg.get("spectral_dropout_rate", 0.25)),
            head_hidden_sizes=_build_head_hidden_sizes_from_str(cfg.get("head_hidden_sizes", "128_32")),
            concat_dropout_rate=float(cfg.get("concat_dropout_rate", 0.25)),
            alpha=float(alpha),
        )
    else:
        gate_in_features = arch_string.split("_")[1]
        return GatedConcatterConfig(
            model_name="GatedConcatter",
            n_filters=[16, 32, 64, 128],
            kernel_sizes=[(100, 3), (15, 10), (10, 3), (5, 2)],
            strides=[(2, 2), (2, 2), (1, 1), (1, 1)],
            raw_dropout_rate=float(cfg.get("raw_dropout_rate", 0.25)),
            paddings=[(25, 1), (5, 2), (5, 1), (1, 1)],
            activation="silu",
            n_spectral_features=16,
            spectral_dropout_rate=float(cfg.get("spectral_dropout_rate", 0.25)),
            head_hidden_sizes=_build_head_hidden_sizes_from_str(cfg.get("head_hidden_sizes", "128_32")),
            concat_dropout_rate=float(cfg.get("concat_dropout_rate", 0.25)),
            gate_in_features=gate_in_features,
        )


def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    network_config = _build_network_from_wandb(cfg)

    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate", 3e-3)),
        weight_decay=None,
        use_cosine_annealing=False,
        cosine_annealing_t_0=5,
        cosine_annealing_t_mult=1,
        cosine_annealing_eta_min=1e-6,
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
        random_seed=int(os.getenv("RANDOM_SEED", 42)),
        batch_size=int(cfg.get("batch_size", 32)),
        max_epochs=30,
        patience=5,
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

    run_config = build_run_config_from_wandb(cfg)

    training_dataset = build_dataset(
        run_config.dataset_config,
        subjects_path=train_subjects_path,
        validation=False,
    )
    validation_dataset = build_dataset(
        run_config.dataset_config,
        subjects_path=val_subjects_path,
        validation=True,
    )
    
    magic_logger = make_logger(wandb_enabled=run_config.log_to_wandb, wandb_init=run_config.wandb_init)

    trained_model = run_single(
        config=run_config,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        logger_sink=magic_logger,
    )

    result = evaluate_with_config(
        model=trained_model,
        dataset=validation_dataset,
        run_config=run_config,
        logger_sink=magic_logger,
        prefix="val",
    )

    pretty_print_per_subject(result.per_subject)

    magic_logger.finish()


if __name__ == "__main__":
    main()