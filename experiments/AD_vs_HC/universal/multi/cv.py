import json
import os
from dotenv import load_dotenv

from core.schemas import MultiDatasetConfig, OptimizerConfig, CriterionConfig, RunConfig
from models.shallow_concatter_se import ShallowConcatterSEConfig
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

def main() -> None:
    model_config = ShallowConcatterSEConfig(
        model_name="ShallowConcatterSE",
        use_se_blocks=True,
        reduction_ratio=4,
        n_filters=[16, 32],
        kernel_sizes=[(40, 2), (8, 5)],
        strides=[(12, 6), (10, 5)],
        paddings=[(5, 0), (1, 1)],
        raw_dropout_rate=0.39811932916850734,
        n_spectral_features=16,
        spectral_hidden_size=128,
        spectral_dropout_rate=0.2809098483832669,
        concat_dropout_rate=0.5642264679590964,
        fusion_hidden_size=256,
        activation="relu",
        gap_length=8,
        raw_norm_type="group",
        spectral_norm_type="none",
        fusion_norm_enabled=False,
    )   
    optimizer_config = OptimizerConfig(
        learning_rate=0.004255107493153422,
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
    dataset_config = MultiDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        dataset_names=["poctep", "hurh", "meg"],
        raw_normalization='channel-subject',
        spectral_normalization='standard',
    )
    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        dataset_config=dataset_config,
        random_seed=42,
        batch_size=32,
        max_epochs=50,
        patience=10,
        min_delta=0.001,
        early_stopping_metric='loss',
        log_to_wandb=True,
        wandb_init={
            "project": "test-da-framework",
        },
    )

    magic_logger = make_logger(wandb_enabled=run_config.log_to_wandb, wandb_init=run_config.wandb_init)
    magic_logger.log_params(run_config.model_dump())

    all_subjects = []
    for dataset_name in dataset_config.dataset_names:
        all_subjects.extend(_read_subjects(SPLITS_JSON_PATH, dataset_name))

    run_cv(
        all_subjects=all_subjects,
        n_folds=N_FOLDS,
        run_config=run_config,
        magic_logger=magic_logger,   
        min_fold_mcc=.20,
    )

if __name__ == "__main__":
    main()