import os
from typing import List

import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold  # type: ignore

from core.builders import build_dataset
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    SpectralDatasetConfig,
)
from core.runner import run as run_single
from core.evaluation import evaluate_with_config, pretty_print_per_subject
from core.logging import make_logger

from models.spectral_net import SpectralNetConfig

from dotenv import load_dotenv
load_dotenv()

def _read_subjects(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def _subject_type(subject_id: str) -> str:
    if subject_id.startswith("ADMIL"):
        return "ADMIL"
    elif subject_id.startswith("ADMOD"):
        return "ADMOD"
    elif subject_id.startswith("HC"):
        return "HC"
    return "UNKNOWN"

def _count_by_category(subjects: List[str]) -> dict[str, int]:
    counts = {"ADMIL": 0, "ADMOD": 0, "HC": 0}
    for subject in subjects:
        category = _subject_type(subject)
        if category in counts:
            counts[category] += 1
    return counts

def main() -> None:
    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5",
    )

    splits_dir = "experiments/AD_vs_HC/combined/spectral/splits"
    train_subjects_path = os.path.join(splits_dir, "training_subjects.txt")
    val_subjects_path = os.path.join(splits_dir, "validation_subjects.txt")

    train_subjects = _read_subjects(train_subjects_path)
    val_subjects = _read_subjects(val_subjects_path)

    all_subjects = train_subjects + val_subjects

    n_folds = 5
    assert len(all_subjects) >= n_folds, "Not enough subjects for the number of folds"

    # Use StratifiedKFold for proper stratified splitting by category
    labels = [_subject_type(sid) for sid in all_subjects]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds: List[List[str]] = []
    for _, test_idx in skf.split(all_subjects, labels):
        folds.append([all_subjects[i] for i in test_idx])

    fold_metrics: List[dict] = []

    model_config = SpectralNetConfig()
    optimizer_config = OptimizerConfig(
        learning_rate=3e-3,
        weight_decay=None,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='fixed',
        pos_weight_value=1.0,
    )

    dataset_config = SpectralDatasetConfig(
        h5_file_path=h5_file_path,
        spectral_normalization='standard',
    )

    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=42,
        batch_size=32,
        max_epochs=50,
        patience=1,
        min_delta=0.001,
        early_stopping_metric='mcc',
        dataset_config=dataset_config,
        log_to_wandb=False,
        wandb_init=None,
    )

    magic_logger = make_logger(wandb_enabled=False, wandb_init=None)

    for fold_idx in range(n_folds):
        val_fold = folds[fold_idx]
        train_fold = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]

        # Log validation subject IDs exactly once per fold
        logger.info(f"Fold {fold_idx+1} validation subjects: {sorted(val_fold)}")

        # Count subjects by category for metrics
        train_counts = _count_by_category(train_fold)
        val_counts = _count_by_category(val_fold)

        logger.info(f"Fold {fold_idx+1}: train={len(train_fold)} subjects ({train_counts}), val={len(val_fold)} subjects ({val_counts})")

        training_dataset = build_dataset(
            dataset_config,
            subjects_list=train_fold,
            validation=False
        )
        validation_dataset = build_dataset(
            dataset_config, 
            subjects_list=val_fold,
            validation=True
        )

        trained_model = run_single(
            config=run_config,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            logger_sink=magic_logger,
        )

        eval_res = evaluate_with_config(
            model=trained_model,
            dataset=validation_dataset,
            run_config=run_config,
            logger_sink=magic_logger,
            prefix=f"fold{fold_idx+1}__val",
        )
        pretty_print_per_subject(eval_res.per_subject, title=f"Fold {fold_idx+1} per-subject")

        fold_metrics.append(eval_res.metrics)

        # Clear separator between folds
        if fold_idx < n_folds - 1:
            logger.info("=" * 40 + f" FOLD {fold_idx+1} " + "=" * 40)

    keys = [
        "final_accuracy", "final_f1", "final_precision", "final_recall", "final_mcc", "final_roc_auc", "final_loss",
        "final_tn", "final_fp", "final_fn", "final_tp"
    ]
    # Compute and log CV summary metrics
    cv_summary_metrics = {}
    logger.info("CV summary (across folds):")
    for key in keys:
        vals = []
        for m in fold_metrics:
            found = [v for k, v in m.items() if k.endswith("/" + key)]
            if found:
                vals.append(float(found[0]))
        if vals:
            vals_array = np.array(vals)
            cv_summary_metrics[f"cv/{key}_min"] = vals_array.min()
            cv_summary_metrics[f"cv/{key}_max"] = vals_array.max()
            cv_summary_metrics[f"cv/{key}_mean"] = vals_array.mean()
            cv_summary_metrics[f"cv/{key}_std"] = vals_array.std()
            logger.info(f"  {key}: min={vals_array.min():.4f}, max={vals_array.max():.4f}, mean={vals_array.mean():.4f}, std={vals_array.std():.4f}")

    # Log CV summary metrics with magic logger
    magic_logger.log_metrics(cv_summary_metrics)


if __name__ == "__main__":
    main()
