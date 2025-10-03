import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from core.schemas import BaseModelConfig
from loguru import logger
from typing import List, Dict, Any
from copy import deepcopy
from core.raw_dataset import RawDataset
from core.eval_model import evaluate_model
from sklearn.model_selection import StratifiedKFold # type: ignore

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


def run_enhanced_cv(
    model: nn.Module,
    config: BaseModelConfig,
    h5_file_path: str,
    included_subjects: List[str],
    n_folds: int,
) -> Dict[str, Any]:
    """
    Enhanced CV that captures comprehensive training metrics per fold,
    similar to what run_sweep.py logs.
    """
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

    logger.info(f"Running enhanced {n_folds}-fold CV over {len(included_subjects)} subjects")

    fold_results: list[Dict[str, Any]] = []

    for fold_idx in range(n_folds):
        val_subjects = folds[fold_idx]
        train_subjects = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]

        logger.info(f"Fold {fold_idx+1}/{n_folds}: train_subjects={len(train_subjects)}, val_subjects={len(val_subjects)}")

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

        # Get training metrics by running training with logging
        training_metrics = _run_training_with_metrics(model, config, training_dataset, validation_dataset)

        # Get comprehensive evaluation metrics
        evaluation_metrics = evaluate_model(model, validation_dataset, batch_size=config.batch_size)

        # Combine training and evaluation metrics for this fold
        fold_result = {
            **training_metrics,
            **evaluation_metrics,
            "fold_info": {
                "fold_idx": fold_idx + 1,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "train_dist": train_dist,
                "val_dist": val_dist,
            }
        }

        fold_results.append(fold_result)

    # Aggregate across folds
    return _aggregate_fold_results(fold_results)


def _run_training_with_metrics(
    model: nn.Module,
    config: BaseModelConfig,
    training_dataset: Dataset,
    validation_dataset: Dataset,
) -> Dict[str, Any]:
    """
    Run training and capture comprehensive metrics similar to run_sweep.py
    """
    from datetime import datetime
    import time
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from sklearn.metrics import matthews_corrcoef, f1_score # type: ignore

    device = _get_device()
    torch.manual_seed(42)
    np.random.seed(42)

    now = datetime.now()
    model = model.to(device)

    train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    if hasattr(config, "pos_weight") and config.pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight))
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    if hasattr(config, "weight_decay") and config.weight_decay is not None:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Track training metrics
    training_history: Dict[str, List[float]] = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
    }

    if not hasattr(config, "early_stopping_metric") or config.early_stopping_metric is None:
        config.early_stopping_metric = 'loss'

    best_val_metric = float('inf') if config.early_stopping_metric == 'loss' else -float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(getattr(config, 'max_epochs', 30)):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()
            outputs = model(data).squeeze(1)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = (outputs > 0.5).float()
            epoch_correct += (predictions == target).sum().item()
            epoch_total += target.size(0)

        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_correct / epoch_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device).float()
                logits = model(data).squeeze(1)
                loss = criterion(logits, target)
                val_loss += loss.item()

                outputs = torch.sigmoid(logits)
                predictions = (outputs > 0.5).float()

                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend(target.cpu().numpy())

                val_correct += (predictions == target).sum().item()
                val_total += target.size(0)

        avg_val_loss = val_loss / len(validation_loader)
        avg_val_accuracy = val_correct / val_total

        train_time = time.time() - epoch_start_time

        # Record metrics
        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(avg_loss)
        training_history["train_accuracy"].append(avg_accuracy)
        training_history["val_loss"].append(avg_val_loss)
        training_history["val_accuracy"].append(avg_val_accuracy)
        training_history["epoch_times"].append(train_time)

        # Early stopping logic
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        if config.early_stopping_metric == 'loss':
            early_stopping_metric = avg_val_loss
        elif config.early_stopping_metric == 'f1':
            early_stopping_metric = f1_score(y_true, y_pred)
        elif config.early_stopping_metric == 'mcc':
            early_stopping_metric = matthews_corrcoef(y_true, y_pred)

        if config.early_stopping_metric == 'loss':
            is_better = early_stopping_metric < best_val_metric - getattr(config, 'min_delta', 0.001)
        else:
            is_better = early_stopping_metric > best_val_metric + getattr(config, 'min_delta', 0.001)

        if is_better:
            best_val_metric = early_stopping_metric
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= getattr(config, 'patience', 5):
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "training_history": training_history,
        "best_epoch": len(training_history["epochs"]) - patience_counter,
        "early_stopping_metric": config.early_stopping_metric,
        "final_train_loss": training_history["train_loss"][-1],
        "final_train_accuracy": training_history["train_accuracy"][-1],
        "final_val_loss": training_history["val_loss"][-1],
        "final_val_accuracy": training_history["val_accuracy"][-1],
    }


def _aggregate_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across folds with comprehensive statistics"""

    # Extract evaluation metrics for aggregation (same as original run_cv)
    selected_keys = [
        "val/final_accuracy",
        "val/final_f1",
        "val/final_precision",
        "val/final_recall",
        "val/final_roc_auc",
        "val/final_mcc",
        "val/tn", "val/fp", "val/fn", "val/tp",
        "val/best_threshold",
    ]

    aggregate: dict[str, dict[str, float]] = {}
    for key in selected_keys:
        values = [fold[key] for fold in fold_results if key in fold]
        if not values:
            continue
        aggregate[key] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    # Aggregate training metrics across folds
    training_aggregates = {
        "best_epoch": {
            "min": min(fold["best_epoch"] for fold in fold_results),
            "max": max(fold["best_epoch"] for fold in fold_results),
            "mean": np.mean([fold["best_epoch"] for fold in fold_results]),
            "std": np.std([fold["best_epoch"] for fold in fold_results]),
        },
        "total_epochs_trained": {
            "min": min(len(fold["training_history"]["epochs"]) for fold in fold_results),
            "max": max(len(fold["training_history"]["epochs"]) for fold in fold_results),
            "mean": np.mean([len(fold["training_history"]["epochs"]) for fold in fold_results]),
        }
    }

    # Stability analysis
    mcc_values = [fold["val/final_mcc"] for fold in fold_results]
    stability_analysis = {
        "mcc_cv_coefficient": float(np.std(mcc_values) / np.mean(mcc_values)) if np.mean(mcc_values) != 0 else 0.0,
        "mcc_range": float(np.max(mcc_values) - np.min(mcc_values)),
        "consistent_performance": all(mcc >= np.mean(mcc_values) - 0.05 for mcc in mcc_values),
    }

    return {
        "folds": fold_results,
        "aggregate": aggregate,
        "training_aggregates": training_aggregates,
        "stability_analysis": stability_analysis,
        "fold_details": [
            fold["fold_info"] for fold in fold_results
        ]
    }
