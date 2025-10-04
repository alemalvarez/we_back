import os
from typing import Literal

import wandb

from core.spectral_dataset import SpectralDataset
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
)
from core.runner import run as run_single
from models.spectral_net import SpectralNetConfig
from dotenv import load_dotenv

load_dotenv()


def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    model_config = SpectralNetConfig(
        model_name="SpectralNet",
        input_size=int(cfg.get("input_size", 16)),
        hidden_1_size=int(cfg.get("hidden_1_size", 16)),
        hidden_2_size=int(cfg.get("hidden_2_size", 16)),
        dropout_rate=float(cfg.get("dropout_rate", 0.25)),
    )

    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate", 3e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)) if cfg.get("weight_decay") is not None else None,
        use_cosine_annealing=bool(cfg.get("use_cosine_annealing", False)),
        cosine_annealing_t_0=int(cfg.get("cosine_annealing_t_0", 5)),
        cosine_annealing_t_mult=int(cfg.get("cosine_annealing_t_mult", 1)),
        cosine_annealing_eta_min=float(cfg.get("cosine_annealing_eta_min", 1e-6)),
    )

    pw_type_str = str(cfg.get("pos_weight_type", "fixed"))
    pw_type_lit: Literal['fixed','multiplied'] = pw_type_str if pw_type_str in ("fixed","multiplied") else "fixed"  # type: ignore[assignment]
    criterion_config = CriterionConfig(
        pos_weight_type=pw_type_lit,
        pos_weight_value=float(cfg.get("pos_weight_value", 1.0)),
    )

    esm_str = str(cfg.get("early_stopping_metric", "mcc"))
    esm_lit: Literal['loss','f1','mcc','kappa'] = esm_str if esm_str in ("loss","f1","mcc","kappa") else "loss"  # type: ignore[assignment]

    norm_str = str(cfg.get("normalization", "standard"))
    norm_lit: Literal['min-max','standard','none'] = norm_str if norm_str in ("min-max","standard","none") else "standard"  # type: ignore[assignment]

    run_config = RunConfig(
        model_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=int(cfg.get("random_seed", 42)),
        batch_size=int(cfg.get("batch_size", 32)),
        max_epochs=int(cfg.get("max_epochs", 50)),
        patience=int(cfg.get("patience", 15)),
        min_delta=float(cfg.get("min_delta", 0.001)),
        early_stopping_metric=esm_lit,
        normalization=norm_lit,
        log_to_wandb=True,
        wandb_init=None,
    )
    return run_config


def main() -> None:
    # Expect a W&B agent to have initialized the run; otherwise, init minimally
    if wandb.run is None:
        wandb.init(project=os.getenv("WANDB_PROJECT", "AD_vs_HC"))

    cfg = wandb.config

    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5",
    )
    train_subjects = os.getenv(
        "TRAIN_SUBJECTS",
        "experiments/AD_vs_HC/combined/spectral/splits/training_subjects.txt",
    )
    val_subjects = os.getenv(
        "VAL_SUBJECTS",
        "experiments/AD_vs_HC/combined/spectral/splits/validation_subjects.txt",
    )

    run_config = build_run_config_from_wandb(cfg)

    training_dataset = SpectralDataset(
        h5_file_path=h5_file_path,
        subjects_txt_path=train_subjects,
        normalize=run_config.normalization,  # type: ignore[arg-type]
    )
    validation_dataset = SpectralDataset(
        h5_file_path=h5_file_path,
        subjects_txt_path=val_subjects,
        normalize=run_config.normalization,  # type: ignore[arg-type]
    )

    run_single(
        config=run_config,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
    )


if __name__ == "__main__":
    main()