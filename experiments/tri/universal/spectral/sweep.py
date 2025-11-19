import json
import os
from dotenv import load_dotenv
import wandb

from core.schemas import SpectralDatasetConfig, OptimizerConfig, CriterionConfig, RunConfig
from models.spectral_net import AdvancedSpectralNetConfig
from core.logging import make_logger
from core.cv import run_cv

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")
SPLITS_JSON_PATH = "experiments/tri/universal/universal_splits_tri.json"
N_FOLDS = 5

def _read_subjects(path: str, dataset_name: str) -> list[str]:
    with open(path, "r") as f:
        splits = json.load(f)
        return splits[dataset_name]["cv_subjects"]

def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    model_config = AdvancedSpectralNetConfig(
        model_name="AdvancedSpectralNet",
        input_size=16,
        hidden_1_size=int(cfg.get("hidden_1_size", 16)),
        hidden_2_size=int(cfg.get("hidden_2_size", 32)),
        dropout_rate=float(cfg.get("dropout_rate", 0.5)),
        add_batch_norm=True,
        activation=cfg.get("activation", "relu"),
    )
    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate", 0.003111076215981144)),
        weight_decay=float(cfg.get("weight_decay", 0.0)) if cfg.get("weight_decay") is not None else None,
        use_cosine_annealing=False,
    )
    criterion_config = CriterionConfig() # nothing to set for tri-class classification!
    dataset_config = SpectralDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        dataset_names=["meg"],
        spectral_normalization="standard",
    )
    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        dataset_config=dataset_config,
        random_seed=int(os.getenv("RANDOM_SEED", 42)),
        batch_size=int(cfg.get("batch_size", 32)),
        max_epochs=75,
        patience=10,
        min_delta=0.001,
        early_stopping_metric='loss',
        log_to_wandb=True,
        wandb_init=None,
        tri_class_it=True,
    )
    return run_config

def main() -> None:
    # Expect a W&B agent to have initialized the run; otherwise, init minimally
    if wandb.run is None:
        wandb.init(project=os.getenv("WANDB_PROJECT", "AD_vs_MCI_vs_AD"))

    cfg = wandb.config

    run_config = build_run_config_from_wandb(cfg)
    magic_logger = make_logger(wandb_enabled=run_config.log_to_wandb, wandb_init=run_config.wandb_init)
    magic_logger.log_params(run_config.model_dump())

    all_subjects = []
    for dataset_name in run_config.dataset_config.dataset_names:
        all_subjects.extend(_read_subjects(SPLITS_JSON_PATH, dataset_name))

    run_cv(
        all_subjects=all_subjects,
        n_folds=N_FOLDS,
        run_config=run_config,
        magic_logger=magic_logger,   
        min_fold_mcc=.20,
    )

    magic_logger.finish()

if __name__ == "__main__":
    main()
