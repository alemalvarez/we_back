from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    matthews_corrcoef,
    cohen_kappa_score,
)
from loguru import logger

from core.logging import Logger


@dataclass
class EvaluationResult:
    metrics: Mapping[str, float]
    best_threshold: float
    per_subject: Optional[Dict[str, Dict[str, int]]] = None

def _unpack_batch(batch: tuple, device: torch.device) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Unpack batch and move to device. Handles both (x, y) and ((x_raw, x_spectral), y) formats."""
    data_or_tuple, target = batch  # type: ignore
    
    if isinstance(data_or_tuple, (tuple, list)):
        data: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = tuple(d.to(device) for d in data_or_tuple)
    else:
        data = data_or_tuple.to(device)
    
    target = target.to(device).float()
    return data, target

def evaluate_dataset(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    criterion: Optional[nn.Module] = None,
    logger_sink: Optional[Logger] = None,
    prefix: str = "val",
    fixed_threshold: Optional[float] = None,

) -> EvaluationResult:
    """Run a comprehensive evaluation on a dataset.

    Computes probabilities, searches a best classification threshold (by MCC),
    builds a confusion matrix, and reports common metrics. If a criterion is
    provided, it also reports the average loss.

    Args:
        model: Trained model (already on the correct device).
        dataset: Dataset to evaluate.
        device: torch device to use.
        batch_size: Dataloader batch size.
        criterion: Optional loss function to compute average loss.
        logger_sink: Optional logger to record scalar metrics.
        prefix: Metric name prefix (e.g., "val" or "test").
        fixed_threshold: Optional fixed threshold to use instead of searching.

    Returns:
        EvaluationResult with a metrics mapping and the best threshold found.
    """

    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_pred_proba_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    subjects: Optional[List[str]] = None
    if hasattr(dataset, "sample_to_subject"):
        subjects = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data, target = _unpack_batch(batch, device)

            logits = model(data).squeeze(1)
            outputs = torch.sigmoid(logits)

            if criterion is not None:
                loss = criterion(logits, target)
                total_loss += float(loss.item())

            proba_np = outputs.cpu().numpy()
            target_np = target.cpu().numpy()
            y_pred_proba_list.extend(proba_np)
            y_true_list.extend(target_np)
            if subjects is not None:
                # Reconstruct per-sample subjects from dataset
                start = batch_idx * loader.batch_size  # type: ignore
                end = start + proba_np.shape[0]
                for i in range(start, end):
                    subjects.append(getattr(dataset, "sample_to_subject")[i])  # type: ignore[index]

    y_pred_proba = np.array(y_pred_proba_list)
    y_true = np.array(y_true_list)

    # Guard against degenerate cases
    if y_true.size == 0:
        raise ValueError("Empty dataset provided to evaluator")

    # Average loss if available
    metrics: dict[str, float] = {}

    # ROC AUC over probabilities
    try:
        metrics[f"{prefix}/final_roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        metrics[f"{prefix}/final_roc_auc"] = float("nan")

    # Threshold search using MCC as objective (or use fixed threshold)
    if fixed_threshold is not None:
        best_threshold = fixed_threshold
    else:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        if thresholds.size == 0:
            best_threshold = 0.5
        else:
            mccs = []
            for t in thresholds:
                y_pred_at_t = (y_pred_proba > t).astype(int)
                try:
                    mccs.append(matthews_corrcoef(y_true, y_pred_at_t))
                except Exception:
                    mccs.append(-1.0)
            best_idx = int(np.argmax(np.array(mccs)))
            best_threshold = float(thresholds[best_idx])

    y_pred = (y_pred_proba > best_threshold).astype(int)

    metrics[f"{prefix}/final_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics[f"{prefix}/final_f1"] = float(f1_score(y_true, y_pred))
    metrics[f"{prefix}/final_precision"] = float(precision_score(y_true, y_pred))
    metrics[f"{prefix}/final_recall"] = float(recall_score(y_true, y_pred))
    metrics[f"{prefix}/final_mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics[f"{prefix}/final_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics[f"{prefix}/final_best_threshold"] = best_threshold

    # Optional per-subject breakdown
    per_subject: Optional[Dict[str, Dict[str, int]]] = None
    if subjects is not None:
        per_subject = {}
        for subj, y_t, y_p in zip(subjects, y_true.tolist(), y_pred.tolist()):
            entry = per_subject.setdefault(subj, {"correct": 0, "wrong": 0})
            if int(y_t) == int(y_p):
                entry["correct"] += 1
            else:
                entry["wrong"] += 1

    # Optional logging to the unified logger
    if logger_sink is not None:
        try:
            logger_sink.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics: {e}")

    return EvaluationResult(metrics=metrics, best_threshold=best_threshold, per_subject=per_subject)


def evaluate_with_config(
    *,
    model: nn.Module,
    dataset: Dataset,
    run_config,
    logger_sink: Optional[Logger] = None,
    prefix: str = "val",
):
    """Convenience wrapper to evaluate without manual device/criterion wiring."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    criterion: Optional[nn.Module] = None
    try:
        from core.builders import build_criterion  # local import to avoid cycles
        from core.runner import _count_pos_neg  # reuse helper
        criterion = build_criterion(run_config.criterion_config, _count_pos_neg(dataset))
    except Exception:
        criterion = None

    return evaluate_dataset(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=run_config.batch_size,
        criterion=criterion,
        logger_sink=logger_sink,
        prefix=prefix,
    )


def pretty_print_per_subject(per_subject: Optional[Dict[str, Dict[str, int]]], *, title: str = "Per-subject results") -> None:
    """Pretty-print per-subject correct/wrong counts to console.

    This intentionally does not log to W&B, since the breakdown isn't suitable
    as scalar metrics. Uses loguru's console logger.
    """
    if not per_subject:
        logger.info("No per-subject breakdown available.")
        return

    logger.info(title)
    logger.info(f"{'Subject':<36}{'Correct':>10}{'Wrong':>10}{'Acc':>8}")
    logger.info("-" * 64)
    # Stable order by subject id/name
    for subject in sorted(per_subject.keys()):
        counts = per_subject[subject]
        correct = int(counts.get("correct", 0))
        wrong = int(counts.get("wrong", 0))
        total = max(correct + wrong, 1)
        acc = correct / total
        if acc > 0.5:
            logger.success(f"{subject:<36}{correct:>10}{wrong:>10}{acc:>8.2f}")
        else:
            logger.info(f"{subject:<36}{correct:>10}{wrong:>10}{acc:>8.2f}")


