from typing import List, Optional, Dict

import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
)

from core.builders import build_dataset
from core.schemas import (
    RunConfig,
)
from core.runner import run as run_single
from core.evaluation import evaluate_with_config, pretty_print_per_subject
from core.logging import Logger

from dotenv import load_dotenv
load_dotenv()


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

def _compute_subject_level_metrics(
    all_subjects_data: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Compute subject-level metrics using majority voting.
    
    Args:
        all_subjects_data: Dict mapping subject_id to {"correct": int, "wrong": int}
        all_subjects: List of all subject IDs to get true labels
        
    Returns:
        Dictionary of subject-level metrics
    """
    y_true_subject = []
    y_pred_subject = []
    
    for subject in sorted(all_subjects_data.keys()):
        counts = all_subjects_data[subject]
        correct = counts.get("correct", 0)
        wrong = counts.get("wrong", 0)
        
        # True label: 0 for HC, 1 for AD (ADMIL or ADMOD)
        true_label = 0 if "HC" in subject else 1
        
        # Predicted label: majority vote
        # If majority of segments match the true label, predict true label
        # Otherwise, predict the opposite
        if correct > wrong:
            pred_label = true_label
        else:
            pred_label = 1 - true_label
        
        y_true_subject.append(true_label)
        y_pred_subject.append(pred_label)
    
    y_true_np = np.array(y_true_subject)
    y_pred_np = np.array(y_pred_subject)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np).ravel()
    
    metrics = {
        "cv/subject_level_tn": float(tn),
        "cv/subject_level_fp": float(fp),
        "cv/subject_level_fn": float(fn),
        "cv/subject_level_tp": float(tp),
        "cv/subject_level_accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "cv/subject_level_f1": float(f1_score(y_true_np, y_pred_np)),
        "cv/subject_level_precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "cv/subject_level_recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "cv/subject_level_mcc": float(matthews_corrcoef(y_true_np, y_pred_np)),
        "cv/subject_level_kappa": float(cohen_kappa_score(y_true_np, y_pred_np)),
    }
    
    # ROC AUC requires probabilities/confidence scores
    # Use the proportion of segments predicted as AD as the confidence score
    try:
        y_score = []
        for subject in sorted(all_subjects_data.keys()):
            counts = all_subjects_data[subject]
            correct = counts.get("correct", 0)
            wrong = counts.get("wrong", 0)
            total = correct + wrong
            
            # True label: 0 for HC, 1 for AD
            true_label = 0 if subject.startswith("HC") else 1
            
            # Proportion of segments that predicted AD
            if true_label == 1:
                # AD subject: correct segments predicted AD
                ad_segments = correct
            else:
                # HC subject: wrong segments predicted AD
                ad_segments = wrong
            
            confidence_ad = ad_segments / total if total > 0 else 0.5
            y_score.append(confidence_ad)
        
        metrics["cv/subject_level_roc_auc"] = float(roc_auc_score(y_true_np, y_score))
    except Exception:
        metrics["cv/subject_level_roc_auc"] = float("nan")
    
    return metrics

def run_cv(
    all_subjects: List[str],
    n_folds: int,
    run_config: RunConfig,
    magic_logger: Logger,
    min_fold_mcc: Optional[float] = .35,
) -> dict:

    assert len(all_subjects) >= n_folds, "Not enough subjects for the number of folds"

    # Use StratifiedKFold for proper stratified splitting by category
    labels = [_subject_type(sid) for sid in all_subjects]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run_config.random_seed)

    folds: List[List[str]] = []
    for _, test_idx in skf.split(all_subjects, labels):
        folds.append([all_subjects[i] for i in test_idx])

    fold_metrics: List[dict] = []
    all_subjects_data: Dict[str, Dict[str, int]] = {}  # Accumulate subject-level data across folds


    for fold_idx in range(n_folds):
        val_fold = folds[fold_idx]
        train_fold = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]

        # Log validation subject IDs exactly once per fold
        logger.info(f"Fold {fold_idx+1} validation subjects: {sorted(val_fold)}")

        # Count subjects by category for metrics
        train_counts = _count_by_category(train_fold)
        val_counts = _count_by_category(val_fold)

        logger.info(f"Fold {fold_idx+1}: train={len(train_fold)} subjects ({train_counts}), val={len(val_fold)} subjects ({val_counts})")

        # Log fold subject counts
        fold_prefix = f"fold{fold_idx+1}"
        magic_logger.log_metrics({
            f"{fold_prefix}/train_subjects": len(train_fold),
            f"{fold_prefix}/val_subjects": len(val_fold),
            f"{fold_prefix}/admil_count": val_counts["ADMIL"],
            f"{fold_prefix}/admod_count": val_counts["ADMOD"],
            f"{fold_prefix}/hc_count": val_counts["HC"],
        })

        training_dataset = build_dataset(
            run_config.dataset_config,
            subjects_list=train_fold,
            validation=False
        )
        validation_dataset = build_dataset(
            run_config.dataset_config, 
            subjects_list=val_fold,
            validation=True
        )

        trained_model = run_single(
            config=run_config,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            logger_sink=magic_logger,
            metric_prefix=fold_prefix,
        )

        eval_res = evaluate_with_config(
            model=trained_model,
            dataset=validation_dataset,
            run_config=run_config,
            logger_sink=magic_logger,
            prefix=f"{fold_prefix}/val",
        )
        pretty_print_per_subject(eval_res.per_subject, title=f"Fold {fold_idx+1} per-subject")

        fold_metrics.append(eval_res.metrics)
        
        # Accumulate per-subject data across folds
        if eval_res.per_subject:
            for subject, counts in eval_res.per_subject.items():
                if subject not in all_subjects_data:
                    all_subjects_data[subject] = {"correct": 0, "wrong": 0}
                all_subjects_data[subject]["correct"] += counts.get("correct", 0)
                all_subjects_data[subject]["wrong"] += counts.get("wrong", 0)

        # Early stopping check based on fold MCC
        if min_fold_mcc is not None:
            mcc_values = [v for k, v in eval_res.metrics.items() if k.endswith("/val/final_mcc")]
            if mcc_values:
                fold_mcc = float(mcc_values[0])
                if fold_mcc < min_fold_mcc:
                    logger.warning(
                        f"Fold {fold_idx+1} MCC ({fold_mcc:.4f}) is below threshold ({min_fold_mcc:.4f}). "
                        f"Stopping CV early."
                    )
                    magic_logger.log_metrics({
                        "cv/early_stopped": True,
                        "cv/early_stopped_at_fold": fold_idx + 1,
                        "cv/early_stopped_mcc": fold_mcc,
                    })
                    return {"early_stopped": True, "fold": fold_idx + 1, "mcc": fold_mcc}

        # Clear separator between folds
        if fold_idx < n_folds - 1:
            logger.info("=" * 40 + f" FOLD {fold_idx+1} DONE" + "=" * 40)

    keys = [
        "final_accuracy", "final_f1", "final_precision", "final_recall", "final_mcc", "final_roc_auc", "final_loss",
        "final_tn", "final_fp", "final_fn", "final_tp", "final_kappa"
    ]
    # Compute and log CV summary metrics
    cv_summary_metrics = {}
    logger.info("CV summary (across folds):")
    for key in keys:
        vals = []
        for m in fold_metrics:
            found = [v for k, v in m.items() if k.endswith("/val/" + key)]
            if found:
                vals.append(float(found[0]))
        if vals:
            vals_array = np.array(vals)
            cv_summary_metrics[f"cv/segment_level_{key}_min"] = vals_array.min()
            cv_summary_metrics[f"cv/segment_level_{key}_max"] = vals_array.max()
            cv_summary_metrics[f"cv/segment_level_{key}_mean"] = vals_array.mean()
            cv_summary_metrics[f"cv/segment_level_{key}_std"] = vals_array.std()
            logger.info(f"  {key}: min={vals_array.min():.4f}, max={vals_array.max():.4f}, mean={vals_array.mean():.4f}, std={vals_array.std():.4f}")

    # Log CV summary metrics with magic logger
    magic_logger.log_metrics(cv_summary_metrics)
    
    # Compute and log subject-level metrics
    if all_subjects_data:
        logger.info("=" * 100)
        pretty_print_per_subject(all_subjects_data, title="Subject-level results across all folds (voting)")
        
        subject_level_metrics = _compute_subject_level_metrics(all_subjects_data)
        
        # Pretty print subject-level confusion matrix
        tn = int(subject_level_metrics["cv/subject_level_tn"])
        fp = int(subject_level_metrics["cv/subject_level_fp"])
        fn = int(subject_level_metrics["cv/subject_level_fn"])
        tp = int(subject_level_metrics["cv/subject_level_tp"])
        
        logger.info("Subject-level Confusion Matrix:")
        logger.info(f"{'':>20} {'Predicted HC':>15} {'Predicted AD':>15}")
        logger.info(f"{'True HC':<20} {tn:>15} {fp:>15}")
        logger.info(f"{'True AD':<20} {fn:>15} {tp:>15}")
        logger.info("")
        
        logger.info("Subject-level Metrics:")
        for key, value in subject_level_metrics.items():
            if "cv/subject_level" in key:
                metric_name = key.replace("cv/subject_level_", "")
                logger.info(f"  {metric_name}: {value:.4f}")
        
        # Log subject-level metrics to magic logger
        magic_logger.log_metrics(subject_level_metrics)
        cv_summary_metrics.update(subject_level_metrics)
    
    return cv_summary_metrics