import json
import os
from dotenv import load_dotenv
import wandb

from core.schemas import OptimizerConfig, CriterionConfig, RunConfig, RawDatasetConfig
from models.squeezer import FlexibleSEConfig
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

# Architecture presets - each is a complete, validated configuration
ARCHITECTURE_PRESETS = {
    # 2-layer architectures (shallow, fewer params)
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
    
    # 3-layer architectures (medium complexity)
    "small_3layer": {
        "n_filters": [16, 32, 64],
        "kernel_sizes": [(40, 2), (8, 5), (5, 2)],
        "strides": [(12, 6), (10, 5), (5, 2)],
        "paddings": [(5, 0), (1, 1), (1, 0)],
    },
    "medium_3layer": {
        "n_filters": [32, 64, 128],
        "kernel_sizes": [(30, 3), (10, 5), (5, 2)],
        "strides": [(10, 2), (8, 4), (4, 2)],
        "paddings": [(8, 1), (2, 1), (1, 1)],
    },
    "balanced_3layer": {
        "n_filters": [24, 48, 96],
        "kernel_sizes": [(50, 2), (15, 3), (8, 2)],
        "strides": [(8, 2), (6, 3), (4, 2)],
        "paddings": [(10, 1), (3, 1), (2, 1)],
    },
    
    # 4-layer architectures (deeper)
    "small_4layer": {
        "n_filters": [16, 32, 64, 128],
        "kernel_sizes": [(50, 3), (15, 5), (8, 3), (4, 2)],
        "strides": [(10, 2), (8, 3), (4, 2), (2, 1)],
        "paddings": [(10, 1), (3, 1), (2, 1), (1, 0)],
    },
    "deep_4layer": {
        "n_filters": [32, 64, 96, 128],
        "kernel_sizes": [(40, 2), (12, 4), (6, 3), (3, 2)],
        "strides": [(12, 2), (8, 2), (4, 2), (2, 1)],
        "paddings": [(8, 1), (2, 1), (1, 1), (1, 0)],
    },
    
    # Alternative patterns
    "wide_shallow": {
        "n_filters": [64, 128],
        "kernel_sizes": [(60, 2), (12, 4)],
        "strides": [(8, 2), (6, 3)],
        "paddings": [(5, 1), (2, 1)],
    },
    "narrow_deep": {
        "n_filters": [16, 24, 32, 48],
        "kernel_sizes": [(35, 4), (12, 4), (8, 2), (4, 2)],
        "strides": [(8, 3), (6, 2), (4, 2), (2, 1)],
        "paddings": [(5, 1), (2, 1), (2, 1), (1, 0)],
    },
}

def _get_architecture_preset(preset_name: str) -> dict:
    """Get architecture parameters from preset name (all presets are pre-validated)."""
    if preset_name not in ARCHITECTURE_PRESETS:
        raise ValueError(f"Unknown architecture preset: {preset_name}. Available: {list(ARCHITECTURE_PRESETS.keys())}")
    
    return ARCHITECTURE_PRESETS[preset_name]

def build_run_config_from_wandb(cfg: wandb.Config) -> RunConfig:  # type: ignore[name-defined]
    # Get architecture from preset
    architecture_preset = cfg.get("architecture", "tiny_2layer")
    preset = _get_architecture_preset(architecture_preset)
    
    n_filters = preset["n_filters"]
    kernel_sizes = preset["kernel_sizes"]
    strides = preset["strides"]
    paddings = preset["paddings"]
    
    # Parse SE block and normalization parameters
    use_se_blocks = bool(cfg.get("use_se_blocks", True))
    reduction_ratio = int(cfg.get("reduction_ratio", 16))
    norm_type = str(cfg.get("norm_type", "batch"))
    
    input_shape = (1000, 68)
    model_config = FlexibleSEConfig(
        model_name="FlexibleSE",
        use_se_blocks=use_se_blocks,
        norm_type=norm_type,
        n_filters=n_filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation=cfg.get("activation", "relu"),
        dropout_rate=float(cfg.get("dropout_rate", 0.4)),
        reduction_ratio=reduction_ratio,
        input_shape=input_shape,
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
    dataset_config = RawDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        dataset_names=["meg"],
        raw_normalization=cfg.get("raw_normalization", "channel-subject"),
        augment=False,
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
