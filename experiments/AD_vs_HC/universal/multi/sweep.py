import json
import os
from dotenv import load_dotenv
import wandb

from core.schemas import OptimizerConfig, CriterionConfig, RunConfig, MultiDatasetConfig
from models.shallow_concatter_se import ShallowConcatterSEConfig, get_architecture_preset
from core.logging import make_logger
from core.cv import run_cv

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")
SPLITS_JSON_PATH = "experiments/AD_vs_HC/universal/universal_splits.json"
N_FOLDS = 5

def _read_subjects(path: str, dataset_name: str) -> list[str]:
    with open(path, "r") as f:
        splits = json.load(f)
        return splits[dataset_name]["cv_subjects"]

# Architecture presets - shallow 2-layer architectures only
ARCHITECTURE_PRESETS = {
    "tiny_2layer": {
        "n_filters": [16, 32],
        "kernel_sizes": [(40, 2), (8, 5)],
        "strides": [(12, 6), (10, 5)],
        "paddings": [(5, 0), (1, 1)],
    },
    "small_2layer": {
        "n_filters": [32, 64],
        "kernel_sizes": [(50, 3), (10, 5)],
        "strides": [(10, 2), (8, 4)],
        "paddings": [(10, 1), (2, 1)],
    },
    "compact_2layer": {
        "n_filters": [16, 32],
        "kernel_sizes": [(30, 4), (15, 3)],
        "strides": [(15, 3), (12, 2)],
        "paddings": [(5, 1), (3, 1)],
    },
    "medium_2layer": {
        "n_filters": [48, 96],
        "kernel_sizes": [(45, 3), (12, 4)],
        "strides": [(10, 2), (8, 3)],
        "paddings": [(8, 1), (2, 1)],
    },
    "wide_2layer": {
        "n_filters": [64, 128],
        "kernel_sizes": [(60, 2), (12, 4)],
        "strides": [(8, 2), (6, 3)],
        "paddings": [(5, 1), (2, 1)],
    },
}

def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    # Get architecture from preset
    architecture_preset = cfg.get("architecture", "tiny_2layer")
    preset = get_architecture_preset(architecture_preset)
    
    n_filters = preset["n_filters"]
    kernel_sizes = preset["kernel_sizes"]
    strides = preset["strides"]
    paddings = preset["paddings"]
    
    # Parse SE block parameters
    use_se_blocks = bool(cfg.get("use_se_blocks", True))
    reduction_ratio = int(cfg.get("reduction_ratio", 16))
    
    # Parse normalization parameters
    raw_norm_type = str(cfg.get("raw_norm_type", "batch"))
    spectral_norm_type = str(cfg.get("spectral_norm_type", "batch"))
    fusion_norm_enabled = bool(cfg.get("fusion_norm_enabled", True))
    
    # Parse dropout rates
    raw_dropout_rate = float(cfg.get("raw_dropout_rate", 0.4))
    spectral_dropout_rate = float(cfg.get("spectral_dropout_rate", 0.3))
    concat_dropout_rate = float(cfg.get("concat_dropout_rate", 0.4))
    
    # Parse spectral branch parameters
    spectral_hidden_size = int(cfg.get("spectral_hidden_size", 64))
    n_spectral_features = 748  # DK features
    
    # Parse fusion parameters
    fusion_hidden_size = int(cfg.get("fusion_hidden_size", 128))
    gap_length = int(cfg.get("gap_length", 4))
    
    # Parse shared parameters
    activation = str(cfg.get("activation", "relu"))
    
    model_config = ShallowConcatterSEConfig(
        model_name="ShallowConcatterSE",
        use_se_blocks=use_se_blocks,
        reduction_ratio=reduction_ratio,
        n_filters=n_filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        raw_norm_type=raw_norm_type,
        raw_dropout_rate=raw_dropout_rate,
        n_spectral_features=n_spectral_features,
        spectral_hidden_size=spectral_hidden_size,
        spectral_norm_type=spectral_norm_type,
        spectral_dropout_rate=spectral_dropout_rate,
        concat_dropout_rate=concat_dropout_rate,
        fusion_hidden_size=fusion_hidden_size,
        fusion_norm_enabled=fusion_norm_enabled,
        activation=activation,
        gap_length=gap_length,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=float(cfg.get("learning_rate", 0.003111076215981144)),
        weight_decay=float(cfg.get("weight_decay", 0.0)) if cfg.get("weight_decay") is not None else None,
        use_cosine_annealing=True,
        cosine_annealing_t_0=7,
        cosine_annealing_t_mult=2,
        cosine_annealing_eta_min=1e-6,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='multiplied',
        pos_weight_value=1.0,
    )
    dataset_config = MultiDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        dataset_names=["poctep"],
        raw_normalization=cfg.get("raw_normalization", "channel-subject"),
        spectral_normalization='standard',
    )
    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        dataset_config=dataset_config,
        random_seed=int(os.getenv("RANDOM_SEED", 42)),
        batch_size=int(cfg.get("batch_size", 32)),
        max_epochs=50,
        patience=10,
        min_delta=0.001,
        early_stopping_metric='loss',
        log_to_wandb=True,
        wandb_init=None
    )
    return run_config

def main() -> None:
    # Expect a W&B agent to have initialized the run; otherwise, init minimally
    if wandb.run is None:
        wandb.init(project=os.getenv("WANDB_PROJECT", "AD_vs_HC"))

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
        min_fold_mcc=.30,
    )

    magic_logger.finish()

if __name__ == "__main__":
    main()
