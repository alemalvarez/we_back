from collections import defaultdict
from collections.abc import Sized
from typing import Dict

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


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


def evaluate_model(
    model: torch.nn.Module,
    evaluation_dataset: Dataset,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Run evaluation on a dataset and print the same metrics computed in training/sweeps.

    Returns a dict with the key metrics for potential reuse.
    """
    assert isinstance(evaluation_dataset, Sized), "Evaluation dataset must be a Sized object"

    device = _get_device()
    model = model.to(device)
    model.eval()

    loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False)

    y_pred_proba_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device).float()
            logits = model(data).squeeze(1)
            probabilities = torch.sigmoid(logits)
            y_pred_proba_list.extend(probabilities.cpu().numpy())
            y_true_list.extend(target.cpu().numpy())

    y_pred_proba = np.array(y_pred_proba_list)
    y_true = np.array(y_true_list)
    logger.info(f"Collected {len(y_true)} predictions from evaluation set")

    # Threshold selection via PR-curve F1, mirroring training
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s)) if f1s.size > 0 else 0
    # sklearn returns thresholds with len = n_points-1; guard for empty
    best_threshold = thresholds[best_idx] if thresholds.size > 0 else 0.5
    logger.info(f"Best threshold: {best_threshold:.4f}")

    y_pred = (y_pred_proba > best_threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    logger.info(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Optional per-subject breakdown if dataset provides mapping
    if hasattr(evaluation_dataset, "sample_to_subject"):
        sample_to_subject = getattr(evaluation_dataset, "sample_to_subject")
        assert len(sample_to_subject) == len(y_true), "sample_to_subject length must match number of samples"
        subject_correct: defaultdict[str, int] = defaultdict(int)
        subject_wrong: defaultdict[str, int] = defaultdict(int)
        for idx, subject in enumerate(sample_to_subject):
            if y_pred[idx] == y_true[idx]:
                subject_correct[subject] += 1
            else:
                subject_wrong[subject] += 1
        for subject in sorted(set(sample_to_subject)):
            logger.info(f"Subject {subject}: correct={subject_correct[subject]}, wrong={subject_wrong[subject]}")

    final_accuracy = float(accuracy_score(y_true, y_pred))
    final_f1 = float(f1_score(y_true, y_pred))
    final_precision = float(precision_score(y_true, y_pred))
    final_recall = float(recall_score(y_true, y_pred))
    final_roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    final_mcc = float(matthews_corrcoef(y_true, y_pred))

    metrics = {
        "val/final_accuracy": final_accuracy,
        "val/final_f1": final_f1,
        "val/final_precision": final_precision,
        "val/final_recall": final_recall,
        "val/final_roc_auc": final_roc_auc,
        "val/final_mcc": final_mcc,
        "val/tn": float(tn),
        "val/fp": float(fp),
        "val/fn": float(fn),
        "val/tp": float(tp),
        "val/best_threshold": float(best_threshold),
    }

    # Print succinctly
    logger.info(
        " | ".join(
            [
                f"acc={final_accuracy:.4f}",
                f"f1={final_f1:.4f}",
                f"prec={final_precision:.4f}",
                f"rec={final_recall:.4f}",
                f"roc_auc={final_roc_auc:.4f}",
                f"mcc={final_mcc:.4f}",
            ]
        )
    )

    return metrics


