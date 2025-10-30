import json
import os
from dotenv import load_dotenv

from core.schemas import SpectralDatasetConfig, OptimizerConfig, CriterionConfig, RunConfig
from models.spectral_net import AdvancedSpectralNetConfig
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
    model_config = AdvancedSpectralNetConfig(
        model_name="AdvancedSpectralNet",
        input_size=16,
        hidden_1_size=32,
        hidden_2_size=16,
        dropout_rate=0.5,
        add_batch_norm=True,
        activation="relu",
    )
    optimizer_config = OptimizerConfig(
        learning_rate=0.003111076215981144,
        weight_decay=0.00027819671966625116,
        use_cosine_annealing=False,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='multiplied',
        pos_weight_value=1.0,
    )
    dataset_config = SpectralDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        dataset_names=["hurh", "poctep"],
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
        min_fold_mcc=.15,
    )

if __name__ == "__main__":
    main()