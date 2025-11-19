from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from core.schemas import RunConfig
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

# Class name mapping for tri-class classification
CLASS_NAMES = {0: "HC", 1: "MCI", 2: "AD"}

@dataclass
class EvaluationResult:
    metrics: Mapping[str, float]
    best_threshold: float
    per_subject: Optional[Dict[str, Dict[str, int]]] = None

def _unpack_batch(batch: tuple, device: torch.device, tri_class_it: bool = False) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Unpack batch and move to device. Handles both (x, y) and ((x_raw, x_spectral), y) formats."""
    data_or_tuple, target = batch  # type: ignore
    
    if isinstance(data_or_tuple, (tuple, list)):
        data: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = tuple(d.to(device) for d in data_or_tuple)
    else:
        data = data_or_tuple.to(device)
    
    target = target.to(device)
    if not tri_class_it:
        target = target.float()
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
    tri_class_it: bool = False,
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
            data, target = _unpack_batch(batch, device, tri_class_it)

            if tri_class_it:
                logits = model(data) # (batch_size, 3)
                outputs = torch.softmax(logits, dim=1)
            else:
                logits = model(data).squeeze(1) # (batch_size, )
                outputs = torch.sigmoid(logits)

            if criterion is not None:
                if tri_class_it:
                    loss = criterion(logits, target.long()) # CrossEntropyLoss
                else:
                    loss = criterion(logits, target) # BCEWithLogitsLoss
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
        if tri_class_it:
            metrics[f"{prefix}/final_roc_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'))
        else:
            metrics[f"{prefix}/final_roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        metrics[f"{prefix}/final_roc_auc"] = float("nan")

    # Threshold search using MCC as objective (or use fixed threshold)
    if tri_class_it:
        # No threshold search for tri-class, use argmax
        y_pred = np.argmax(y_pred_proba, axis=1)
        best_threshold = float('nan')
    elif fixed_threshold is not None:
        best_threshold = fixed_threshold
        y_pred = (y_pred_proba > best_threshold).astype(int)
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
    
    if tri_class_it:
        # Macro-averaged metrics
        metrics[f"{prefix}/final_f1"] = float(f1_score(y_true, y_pred, average='macro'))
        metrics[f"{prefix}/final_precision"] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics[f"{prefix}/final_recall"] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Per-class metrics with human-readable names
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        for class_idx, class_name in CLASS_NAMES.items():
            metrics[f"{prefix}/final_f1_{class_name}"] = float(f1_per_class[class_idx])
            metrics[f"{prefix}/final_precision_{class_name}"] = float(precision_per_class[class_idx])
            metrics[f"{prefix}/final_recall_{class_name}"] = float(recall_per_class[class_idx])
    else:
        metrics[f"{prefix}/final_f1"] = float(f1_score(y_true, y_pred))
        metrics[f"{prefix}/final_precision"] = float(precision_score(y_true, y_pred))
        metrics[f"{prefix}/final_recall"] = float(recall_score(y_true, y_pred))
    
    metrics[f"{prefix}/final_mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics[f"{prefix}/final_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    
    if not tri_class_it:
        metrics[f"{prefix}/final_best_threshold"] = best_threshold

    # Optional per-subject breakdown
    per_subject: Optional[Dict[str, Dict[str, int]]] = None
    if subjects is not None:
        per_subject = {}
        for subj, y_t, y_p in zip(subjects, y_true.tolist(), y_pred.tolist()):
            y_t_int, y_p_int = int(y_t), int(y_p)
            if subj not in per_subject:
                per_subject[subj] = {"true_label": y_t_int}
                # Initialize prediction counters
                for class_idx in range(3 if tri_class_it else 2):
                    per_subject[subj][f"pred_{CLASS_NAMES[class_idx]}"] = 0
            # Increment prediction count
            per_subject[subj][f"pred_{CLASS_NAMES[y_p_int]}"] += 1

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
    run_config: RunConfig,
    logger_sink: Optional[Logger] = None,
    prefix: str = "val",
):
    """Convenience wrapper to evaluate without manual device/criterion wiring."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    criterion: Optional[nn.Module] = None
    try:
        from core.builders import build_criterion  # local import to avoid cycles
        from core.runner import _count_classes  # reuse helper
        criterion = build_criterion(run_config.criterion_config, _count_classes(dataset, tri_class=run_config.tri_class_it), tri_class_it=run_config.tri_class_it)
        criterion = criterion.to(device)
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
        tri_class_it=run_config.tri_class_it
    )


def pretty_print_per_subject(per_subject: Optional[Dict[str, Dict[str, int]]], *, title: str = "Per-subject results", tri_class: bool = False) -> None:
    """Pretty-print per-subject prediction distribution to console.

    This intentionally does not log to W&B, since the breakdown isn't suitable
    as scalar metrics. Uses loguru's console logger.
    """
    if not per_subject:
        logger.info("No per-subject breakdown available.")
        return

    logger.info(title)
    
    # Determine if using new format
    first_entry = next(iter(per_subject.values()))
    is_new_format = "true_label" in first_entry
    
    if not is_new_format:
        # Legacy format fallback
        logger.info(f"{'Subject':<36}{'Correct':>10}{'Wrong':>10}{'Acc':>8}")
        logger.info("-" * 64)
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
        return
    
    # New format with per-class predictions
    if tri_class:
        logger.info(f"{'Subject':<36}{'HC':>10}{'MCI':>10}{'AD':>10}{'Acc':>8}")
        logger.info("-" * 74)
    else:
        logger.info(f"{'Subject':<36}{'HC':>10}{'AD':>10}{'Acc':>8}")
        logger.info("-" * 64)
    
    for subject in sorted(per_subject.keys()):
        counts = per_subject[subject]
        true_label = counts["true_label"]
        true_class = CLASS_NAMES[true_label]
        
        if tri_class:
            hc = counts["pred_HC"]
            mci = counts["pred_MCI"]
            ad = counts["pred_AD"]
            total = hc + mci + ad
            correct = counts[f"pred_{true_class}"]
            acc = correct / total if total > 0 else 0
            row = f"{subject:<36}{hc:>10}{mci:>10}{ad:>10}{acc:>8.2f}"
        else:
            hc = counts["pred_HC"]
            ad = counts["pred_AD"]
            total = hc + ad
            correct = counts[f"pred_{true_class}"]
            acc = correct / total if total > 0 else 0
            row = f"{subject:<36}{hc:>10}{ad:>10}{acc:>8.2f}"
        
        if acc > 0.5:
            logger.success(row)
        else:
            logger.info(row)


