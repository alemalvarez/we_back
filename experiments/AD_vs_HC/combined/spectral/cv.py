import os
from typing import List

from loguru import logger
from sklearn.model_selection import StratifiedKFold  # type: ignore

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

def _subject_type(subject_id: str) -> str:
    if subject_id.startswith("ADMIL"):
        return "ADMIL"
    elif subject_id.startswith("ADMOD"):
        return "ADMOD"
    elif subject_id.startswith("HC"):
        return "HC"
    return "UNKNOWN"

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

    for fold_idx in range(n_folds):
        val_fold = folds[fold_idx]
        train_fold = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]

        logger.info(f"Fold {fold_idx+1}: train={len(train_fold)} subjects, val={len(val_fold)} subjects")

        run_config = RunConfig(
            network_config=model_config,
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
            subjects_list=train_fold,
            normalize=run_config.normalization,
        )
        validation_dataset = SpectralDataset(
            h5_file_path=h5_file_path,
            subjects_list=val_fold,
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

    keys = [
        "final_accuracy", "final_f1", "final_precision", "final_recall", "final_mcc", "final_roc_auc", "final_loss"
    ]
    summary = {}
    for key in keys:
        vals = []
        for m in fold_metrics:
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
