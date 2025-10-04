import glob
import os
from typing import List

from loguru import logger

from core.spectral_dataset import SpectralDataset
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
)
from core.runner import run as run_single
from core.evaluation import evaluate_with_config, pretty_print_per_subject

from models.spectral_net import SpectralNetConfig

from dotenv import load_dotenv
load_dotenv()

def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5",
    )

    splits_dir = "experiments/AD_vs_HC/combined/spectral/splits"
    train_files = sorted(glob.glob(os.path.join(splits_dir, "training_subjects_fold*.txt")))
    val_files = sorted(glob.glob(os.path.join(splits_dir, "validation_subjects_fold*.txt")))
    assert len(train_files) == len(val_files) and len(train_files) > 0, "No CV folds found"

    model_config = SpectralNetConfig()
    optimizer_config = OptimizerConfig(
        learning_rate=3e-3,
        weight_decay=None,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='fixed',
        pos_weight_value=1.0,
    )

    fold_metrics: List[dict] = []

    for fold_idx, (train_path, val_path) in enumerate(zip(train_files, val_files)):
        logger.info(f"Fold {fold_idx+1}: train={os.path.basename(train_path)}, val={os.path.basename(val_path)}")

        train_subjects = train_path
        val_subjects = val_path

        run_config = RunConfig(
            model_config=model_config,
            optimizer_config=optimizer_config,
            criterion_config=criterion_config,
            random_seed=42,
            batch_size=32,
            max_epochs=50,
            patience=15,
            min_delta=0.001,
            early_stopping_metric='mcc',
            normalization='standard',
            log_to_wandb=False,
            wandb_init=None,
        )

        training_dataset = SpectralDataset(
            h5_file_path=h5_file_path,
            subjects_txt_path=train_subjects,
            normalize=run_config.normalization,
        )
        validation_dataset = SpectralDataset(
            h5_file_path=h5_file_path,
            subjects_txt_path=val_subjects,
            normalize=run_config.normalization,
        )

        trained_model = run_single(
            config=run_config,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
        )

        eval_res = evaluate_with_config(
            model=trained_model,
            dataset=validation_dataset,
            run_config=run_config,
            logger_sink=None,
            prefix=f"fold{fold_idx+1}__val",
        )
        pretty_print_per_subject(eval_res.per_subject, title=f"Fold {fold_idx+1} per-subject")

        fold_metrics.append(dict(eval_res.metrics))

    # Aggregate: simple mean across folds for common metrics
    keys = [
        "final_accuracy", "final_f1", "final_precision", "final_recall", "final_mcc", "final_roc_auc", "final_loss"
    ]
    summary = {}
    for key in keys:
        vals = []
        for m in fold_metrics:
            # find prefixed key in this fold
            found = [v for k, v in m.items() if k.endswith("/" + key)]
            if found:
                vals.append(float(found[0]))
        if vals:
            summary[key] = sum(vals) / len(vals)

    logger.info("CV summary (mean across folds):")
    for k, v in summary.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()


