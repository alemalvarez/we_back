import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from core.schemas import BaseModelConfig
from loguru import logger

from typing import List
from copy import deepcopy
from core.raw_dataset import RawDataset
from core.train_model import train_model
from core.eval_model import evaluate_model
from sklearn.model_selection import StratifiedKFold  # type: ignore

def _subject_type(subject_id: str) -> str:
    if subject_id.startswith("ADMIL"):
        return "ADMIL"
    elif subject_id.startswith("ADMOD"):
        return "ADMOD"
    elif subject_id.startswith("HC"):
        return "HC"
    return "UNKNOWN"

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.success("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.success("Using MPS")
        return torch.device("mps")
    else:
        logger.info("Using CPU...")
        return torch.device("cpu")  

def _count_pos_neg(dataset: Dataset) -> tuple[int, int]:
    # Assumes dataset[i][1] is the label and is 0 or 1 or convertible to int
    pos = 0
    neg = 0
    for i in range(len(dataset)): # type: ignore
        label = dataset[i][1]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if int(label) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg



def run_cv(
    model: nn.Module,
    config: BaseModelConfig,
    h5_file_path: str,
    included_subjects: List[str], # the point is leaving subjects out (test)
    n_folds: int,
):
    assert n_folds >= 2, "n_folds must be at least 2"
    assert len(included_subjects) >= n_folds, "n_folds cannot exceed number of included_subjects"

    # Use StratifiedKFold for proper stratified splitting by category
    labels = [_subject_type(sid) for sid in included_subjects]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_seed)

    folds: list[list[str]] = []
    for _, test_idx in skf.split(included_subjects, labels):
        folds.append([included_subjects[i] for i in test_idx])

    original_state = deepcopy(model.state_dict())

    normalize = getattr(config, "normalize", "sample-channel")
    augment = bool(getattr(config, "augment", False))
    augment_prob_neg = float(getattr(config, "augment_prob_neg", 0.5))
    augment_prob_pos = float(getattr(config, "augment_prob_pos", 0.0))
    noise_std = float(getattr(config, "noise_std", 0.1))

    logger.info(f"Running {n_folds}-fold CV over {len(included_subjects)} subjects")

    metrics_per_fold: list[dict[str, float]] = []

    for fold_idx in range(n_folds):
        val_subjects = folds[fold_idx]
        train_subjects = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]

        logger.info(f"Fold {fold_idx+1}/{n_folds}: train_subjects={len(train_subjects)}, val_subjects={len(val_subjects)}")

        logger.debug(f"train_subjects: {train_subjects}")
        logger.debug(f"val_subjects: {val_subjects}")

        # Log category distribution for train and val
        def _dist(subjs: list[str]) -> dict[str, tuple[int, float]]:
            total = len(subjs)
            counts = {"ADMIL": 0, "ADMOD": 0, "HC": 0}
            for s in subjs:
                t = _subject_type(s)
                if t in counts:
                    counts[t] += 1
            return {k: (v, (v / total * 100.0) if total > 0 else 0.0) for k, v in counts.items()}

        train_dist = _dist(train_subjects)
        val_dist = _dist(val_subjects)
        logger.info(
            "Train distribution | "
            + ", ".join([f"{k}={train_dist[k][0]} ({train_dist[k][1]:.1f}%)" for k in ["ADMIL", "ADMOD", "HC"]])
        )
        logger.info(
            "Val distribution   | "
            + ", ".join([f"{k}={val_dist[k][0]} ({val_dist[k][1]:.1f}%)" for k in ["ADMIL", "ADMOD", "HC"]])
        )

        training_dataset = RawDataset(
            h5_file_path=h5_file_path,
            subjects_txt_path="unused",
            normalize=normalize,  # type: ignore[arg-type]
            augment=augment,
            augment_prob=(augment_prob_neg, augment_prob_pos),
            noise_std=noise_std,
            subjects_list=train_subjects,
        )

        validation_dataset = RawDataset(
            h5_file_path=h5_file_path,
            subjects_txt_path="unused",
            normalize=normalize,  # type: ignore[arg-type]
            augment=False,
            subjects_list=val_subjects,
        )

        model.load_state_dict(original_state)
        _ = train_model(model, config, training_dataset, validation_dataset)

        fold_metrics = evaluate_model(model, validation_dataset, batch_size=config.batch_size)
        metrics_per_fold.append(fold_metrics)

    # Aggregate across folds
    selected_keys = [
        "val/final_accuracy",
        "val/final_f1",
        "val/final_precision",
        "val/final_recall",
        "val/final_roc_auc",
        "val/final_mcc",
    ]

    aggregate: dict[str, dict[str, float]] = {}
    for key in selected_keys:
        values = [m[key] for m in metrics_per_fold if key in m]
        if not values:
            continue
        aggregate[key] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    for key, stats in aggregate.items():
        logger.info(f"CV {key}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    return {
        "folds": metrics_per_fold,
        "aggregate": aggregate,
        "fold_details": [
            {
                "train_subjects": [s for i, fold in enumerate(folds) if i != fold_idx for s in fold],
                "val_subjects": folds[fold_idx]
            }
            for fold_idx in range(n_folds)
        ]
    }