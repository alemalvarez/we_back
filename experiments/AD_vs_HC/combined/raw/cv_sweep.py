from core.raw_dataset import RawDataset
from core.model_playground import create_model_from_wandb_config
from core.enhanced_run_cv import run_enhanced_cv
from loguru import logger
import wandb
from dotenv import load_dotenv
import os
import sys
import numpy as np
from typing import Dict, Any

load_dotenv()

WANDB_PROJECT = "AD_vs_HC_cv_sweep"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")

def main():
    """Main function for cross-validated sweep"""
    model_type = None
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        logger.info(f"Using model type: {model_type}")

    with wandb.init(project=WANDB_PROJECT) as run:
        config = run.config

        # Set seeds for reproducibility
        import torch
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # Create full dataset to get all subjects for CV
        full_dataset = RawDataset(
            h5_file_path=H5_FILE_PATH,
            subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
            normalize=getattr(config, "normalize", "sample-channel"),
            augment=False  # No augmentation for CV consistency
        )

        # Get all unique subjects
        all_subjects = sorted(list(set(full_dataset.sample_to_subject)))
        logger.info(f"Running 5-fold CV over {len(all_subjects)} subjects")

        # Log dataset info
        wandb.log({
            "dataset/total_subjects": len(all_subjects),
            "dataset/total_samples": len(full_dataset),
            "dataset/subject_types": {
                "ADMIL": sum(1 for s in all_subjects if s.startswith("ADMIL")),
                "ADMOD": sum(1 for s in all_subjects if s.startswith("ADMOD")),
                "HC": sum(1 for s in all_subjects if s.startswith("HC")),
            }
        })

        # Create model
        model = create_model_from_wandb_config(config, model_type)

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "model/class_name": model.__class__.__name__,
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
        })

        # Run 5-fold CV
        cv_results = run_enhanced_cv(
            model=model,
            config=config,
            h5_file_path=H5_FILE_PATH,
            included_subjects=all_subjects,
            n_folds=5
        )

        # Log comprehensive CV metrics
        log_comprehensive_cv_metrics(run, cv_results)

def log_comprehensive_cv_metrics(run: wandb.Run, cv_results: Dict[str, Any]):
    """Log comprehensive CV metrics with the same level of detail as run_sweep.py"""

    folds_data = cv_results["folds"]

    # Extract MCC values for optimization
    fold_mccs = [fold["val/final_mcc"] for fold in folds_data]
    avg_mcc = np.mean(fold_mccs)
    std_mcc = np.std(fold_mccs)

    # PRIMARY OPTIMIZATION METRIC
    wandb.log({"cv/avg_mcc": avg_mcc})

    # Overall CV summary metrics
    wandb.log({
        "cv/summary_avg_mcc": avg_mcc,
        "cv/summary_std_mcc": std_mcc,
        "cv/summary_mcc_range": np.max(fold_mccs) - np.min(fold_mccs),
        "cv/summary_best_fold_mcc": np.max(fold_mccs),
        "cv/summary_worst_fold_mcc": np.min(fold_mccs),
        "cv/summary_mcc_coefficient_of_variation": std_mcc / avg_mcc if avg_mcc != 0 else 0.0,
    })

    # Per-fold comprehensive metrics (equivalent to what run_sweep.py logs)
    for fold_idx, fold_metrics in enumerate(folds_data):
        fold_prefix = f"fold_{fold_idx+1}"

        # Core final metrics (same as run_sweep.py)
        wandb.log({
            f"{fold_prefix}/final_accuracy": fold_metrics["val/final_accuracy"],
            f"{fold_prefix}/final_f1": fold_metrics["val/final_f1"],
            f"{fold_prefix}/final_precision": fold_metrics["val/final_precision"],
            f"{fold_prefix}/final_recall": fold_metrics["val/final_recall"],
            f"{fold_prefix}/final_roc_auc": fold_metrics["val/final_roc_auc"],
            f"{fold_prefix}/final_mcc": fold_metrics["val/final_mcc"],
        })

        # Confusion matrix components (same as run_sweep.py)
        wandb.log({
            f"{fold_prefix}/tn": fold_metrics["val/tn"],
            f"{fold_prefix}/fp": fold_metrics["val/fp"],
            f"{fold_prefix}/fn": fold_metrics["val/fn"],
            f"{fold_prefix}/tp": fold_metrics["val/tp"],
            f"{fold_prefix}/best_threshold": fold_metrics["val/best_threshold"],
        })

        # Per-subject metrics if available (same as run_sweep.py)
        if "per_subject_metrics" in fold_metrics:
            for subject, subj_metrics in fold_metrics["per_subject_metrics"].items():
                wandb.log({
                    f"{fold_prefix}/subject_{subject}/correct_segments": subj_metrics["correct_segments"],
                    f"{fold_prefix}/subject_{subject}/wrong_segments": subj_metrics["wrong_segments"],
                    f"{fold_prefix}/subject_{subject}/accumulated_loss": subj_metrics["accumulated_loss"],
                    f"{fold_prefix}/subject_{subject}/mean_loss": subj_metrics["mean_loss"],
                })

    # Aggregated metrics across all folds (same structure as run_cv.py but more comprehensive)
    aggregate = cv_results["aggregate"]

    # Core metrics aggregation
    for metric_key, stats in aggregate.items():
        metric_name = metric_key.replace("val/final_", "").replace("val/", "")
        wandb.log({
            f"cv/{metric_name}_mean": stats["mean"],
            f"cv/{metric_name}_std": stats["std"],
            f"cv/{metric_name}_min": stats["min"],
            f"cv/{metric_name}_max": stats["max"],
        })

    # Stability analysis metrics
    if "stability_analysis" in cv_results:
        stability = cv_results["stability_analysis"]
        wandb.log({
            "cv/stability_mcc_cv_coefficient": stability["mcc_cv_coefficient"],
            "cv/stability_mcc_range": stability["mcc_range"],
            "cv/stability_consistent_performance": float(stability["consistent_performance"]),
        })

    # Fold distribution analysis (log subject counts per fold)
    for fold_idx, fold_detail in enumerate(cv_results["fold_details"]):
        fold_prefix = f"fold_{fold_idx+1}"
        train_subjects = fold_detail["train_subjects"]
        val_subjects = fold_detail["val_subjects"]

        # Count subject types in train/val splits
        train_counts = {
            "ADMIL": sum(1 for s in train_subjects if s.startswith("ADMIL")),
            "ADMOD": sum(1 for s in train_subjects if s.startswith("ADMOD")),
            "HC": sum(1 for s in train_subjects if s.startswith("HC")),
        }
        val_counts = {
            "ADMIL": sum(1 for s in val_subjects if s.startswith("ADMIL")),
            "ADMOD": sum(1 for s in val_subjects if s.startswith("ADMOD")),
            "HC": sum(1 for s in val_subjects if s.startswith("HC")),
        }

        wandb.log({
            f"{fold_prefix}/train_subjects_count": len(train_subjects),
            f"{fold_prefix}/val_subjects_count": len(val_subjects),
            f"{fold_prefix}/train_ADMIL": train_counts["ADMIL"],
            f"{fold_prefix}/train_ADMOD": train_counts["ADMOD"],
            f"{fold_prefix}/train_HC": train_counts["HC"],
            f"{fold_prefix}/val_ADMIL": val_counts["ADMIL"],
            f"{fold_prefix}/val_ADMOD": val_counts["ADMOD"],
            f"{fold_prefix}/val_HC": val_counts["HC"],
        })

    # Summary table for easy comparison (log as nested dict)
    summary_table = {
        "fold_metrics": [
            {
                "fold": i+1,
                "mcc": fold["val/final_mcc"],
                "f1": fold["val/final_f1"],
                "accuracy": fold["val/final_accuracy"],
                "precision": fold["val/final_precision"],
                "recall": fold["val/final_recall"],
                "roc_auc": fold["val/final_roc_auc"],
            }
            for i, fold in enumerate(folds_data)
        ],
        "aggregate_stats": {
            "mcc": aggregate["val/final_mcc"],
            "f1": aggregate["val/final_f1"],
            "accuracy": aggregate["val/final_accuracy"],
        }
    }

    # Log summary as a table artifact for W&B
    import pandas as pd # type: ignore

    # Create summary DataFrame
    fold_df = pd.DataFrame(summary_table["fold_metrics"])
    fold_table = wandb.Table(dataframe=fold_df)
    wandb.log({"cv/fold_comparison_table": fold_table})

    # Log aggregate stats as a separate table
    agg_data = []
    for metric, stats in aggregate.items():
        if "val/final_" in metric:
            metric_name = metric.replace("val/final_", "")
            agg_data.append({
                "metric": metric_name,
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"]
            })

    agg_df = pd.DataFrame(agg_data)
    agg_table = wandb.Table(dataframe=agg_df)
    wandb.log({"cv/aggregate_metrics_table": agg_table})

    # Final comprehensive logging message
    logger.success(f"CV completed. Average MCC: {avg_mcc:.4f} ± {std_mcc:.4f}")
    logger.info(f"MCC range: [{np.min(fold_mccs):.4f}, {np.max(fold_mccs):.4f}]")

if __name__ == "__main__":
    main()
