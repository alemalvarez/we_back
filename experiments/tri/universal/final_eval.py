import json
import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from loguru import logger

from core.schemas import OptimizerConfig, CriterionConfig, RunConfig, MultiDatasetConfig
from models.shallow_concatter_se import ShallowConcatterSEConfig
from core.logging import make_logger, Logger
from core.builders import build_dataset
from core.runner import run as run_single
from core.evaluation import evaluate_dataset, pretty_print_per_subject
import torch
from sklearn.model_selection import train_test_split  # type: ignore

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")
SPLITS_JSON_PATH = "experiments/tri/universal/universal_splits_tri.json"

# Dataset categories
CATEGORIES = ["poctep", "hurh", "meg", "eeg", "all"]


def _read_json_subjects(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Read the splits JSON and return all subjects organized by dataset."""
    with open(path, "r") as f:
        return json.load(f)


def _subject_type(subject_id: str) -> str:
    """Get subject type for stratification."""
    if "ADMIL" in subject_id:
        return "ADMIL"
    elif "ADMOD" in subject_id:
        return "ADMOD"
    elif "HC" in subject_id:
        return "HC"
    elif "AD" in subject_id:
        return "AD"
    return "UNKNOWN"


def get_category_subjects(splits_data: Dict[str, Dict[str, List[str]]], category: str) -> Tuple[List[str], List[str]]:
    """Get cv_subjects and test_subjects for a given category.
    
    Args:
        splits_data: The parsed JSON data
        category: One of 'poctep', 'hurh', 'meg', 'eeg', 'all'
    
    Returns:
        Tuple of (cv_subjects, test_subjects)
    """
    if category in ["poctep", "hurh", "meg"]:
        # Direct from JSON
        return (
            splits_data[category]["cv_subjects"],
            splits_data[category]["test_subjects"]
        )
    elif category == "eeg":
        # Combine poctep + hurh
        cv = splits_data["poctep"]["cv_subjects"] + splits_data["hurh"]["cv_subjects"]
        test = splits_data["poctep"]["test_subjects"] + splits_data["hurh"]["test_subjects"]
        return cv, test
    elif category == "all":
        # Combine everything
        cv = (
            splits_data["poctep"]["cv_subjects"] + 
            splits_data["hurh"]["cv_subjects"] + 
            splits_data["meg"]["cv_subjects"]
        )
        test = (
            splits_data["poctep"]["test_subjects"] + 
            splits_data["hurh"]["test_subjects"] + 
            splits_data["meg"]["test_subjects"]
        )
        return cv, test
    else:
        raise ValueError(f"Unknown category: {category}")


def train_and_evaluate(
    training_category: str,
    run_config: RunConfig,
    magic_logger: Logger,
    val_split: float = 0.2,
) -> None:
    """Train a model on cv_subjects from training_category and evaluate on all test sets.
    
    Args:
        training_category: Category to train on ('poctep', 'hurh', 'meg', 'eeg', 'all')
        run_config: Configuration for training
        magic_logger: Logger instance for W&B
        val_split: Fraction of cv_subjects to use for validation
    """
    logger.info("=" * 100)
    logger.info(f"TRAINING ON CATEGORY: {training_category.upper()}")
    logger.info("=" * 100)
    
    # Load all subjects
    splits_data = _read_json_subjects(SPLITS_JSON_PATH)
    
    # Get cv_subjects for training category
    cv_subjects, _ = get_category_subjects(splits_data, training_category)
    
    logger.info(f"Total cv_subjects for {training_category}: {len(cv_subjects)}")
    
    # Split cv_subjects into train/val for early stopping
    labels = [_subject_type(s) for s in cv_subjects]
    train_subjects, val_subjects = train_test_split(
        cv_subjects,
        test_size=val_split,
        random_state=run_config.random_seed,
        stratify=labels,
    )
    
    logger.info(f"Split into train={len(train_subjects)}, val={len(val_subjects)}")
    
    # Build training dataset
    training_dataset = build_dataset(
        run_config.dataset_config,
        subjects_list=train_subjects,
        validation=False,
        tri_class_it=run_config.tri_class_it,
    )
    
    # Extract normalization stats
    train_norm_stats = getattr(training_dataset, 'norm_stats', None)
    
    # Build validation dataset
    validation_dataset = build_dataset(
        run_config.dataset_config,
        subjects_list=val_subjects,
        validation=True,
        norm_stats=train_norm_stats,
        tri_class_it=run_config.tri_class_it,
    )
    
    # Train model
    logger.info("Starting training...")
    trained_model = run_single(
        config=run_config,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        logger_sink=magic_logger,
        metric_prefix=f"train_{training_category}",
    )
    
    logger.info("Training complete. Determining optimal threshold on validation set...")
    
    # Device setup for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Evaluate on validation set to get optimal threshold
    val_result = evaluate_dataset(
        model=trained_model,
        dataset=validation_dataset,
        device=device,
        batch_size=run_config.batch_size,
        tri_class_it=run_config.tri_class_it,
    )
    optimal_threshold = val_result.best_threshold
    logger.success(f"Optimal threshold from validation: {optimal_threshold:.4f}")
    
    logger.info("Starting cross-domain evaluation with fixed threshold...")
    logger.info("=" * 100)
    
    # Evaluate on all test sets
    for test_category in CATEGORIES:
        logger.info(f"\nEvaluating on {test_category.upper()} test set...")
        
        _, test_subjects = get_category_subjects(splits_data, test_category)
        
        if not test_subjects:
            logger.warning(f"No test subjects for {test_category}, skipping...")
            continue
        
        logger.info(f"Test subjects count: {len(test_subjects)}")
        
        # Build test dataset with training normalization stats
        test_dataset = build_dataset(
            run_config.dataset_config,
            subjects_list=test_subjects,
            validation=True,
            norm_stats=train_norm_stats,
            tri_class_it=run_config.tri_class_it,
        )
        
        # Evaluate with fixed threshold
        eval_result = evaluate_dataset(
            model=trained_model,
            dataset=test_dataset,
            device=device,
            batch_size=run_config.batch_size,
            logger_sink=magic_logger,
            prefix=f"test_{test_category}",
            fixed_threshold=optimal_threshold,
            tri_class_it=run_config.tri_class_it,
        )
        
        # Pretty print per-subject results
        pretty_print_per_subject(
            eval_result.per_subject,
            title=f"Trained on {training_category.upper()} → Test on {test_category.upper()}",
            tri_class=run_config.tri_class_it,
        )
        
        # Log summary
        logger.info(f"Results for {training_category} → {test_category}:")
        for key, value in eval_result.metrics.items():
            metric_name = key.split('/')[-1]
            if not metric_name.startswith('final_best'):
                logger.info(f"  {metric_name}: {value:.4f}")
        
        logger.info("-" * 80)
    
    logger.info("=" * 100)
    logger.info(f"COMPLETED EVALUATION FOR TRAINING CATEGORY: {training_category.upper()}")
    logger.info("=" * 100)


def main() -> None:
    """Main entry point."""

    # Configure model with requested hyperparameters (from command-line options)
    # --activation=leaky_relu --architecture=balanced_3layer --batch_size=64 --dropout_rate=0.25047434239185595 --learning_rate=0.005400434790402595 --norm_type=group --raw_normalization=control-global --reduction_ratio=32 --use_se_blocks=False --weight_decay=0.0004147760689219528

    model_config = ShallowConcatterSEConfig(
        model_name="ShallowConcatterSE",
        use_se_blocks=True,
        reduction_ratio=16,
        n_filters=[16, 32],  # "compact_2layer" preset from sweep.py
        kernel_sizes=[(20, 5), (5, 3)],
        strides=[(10, 5), (5, 3)],
        paddings=[(2, 1), (1, 1)],
        raw_norm_type="batch",
        raw_dropout_rate=0.4516341906899774,
        n_spectral_features=16,
        spectral_hidden_size=128,
        spectral_norm_type="none",
        spectral_dropout_rate=0.2606536276320501,
        concat_dropout_rate=0.266882556505772,
        fusion_hidden_size=128,
        fusion_norm_enabled=True,
        activation="leaky_relu",
        gap_length=4,
    )

    optimizer_config = OptimizerConfig(
        learning_rate=0.007057721815610606,
        weight_decay=0.00021286130098368837,
        use_cosine_annealing=False,
    )

    criterion_config = CriterionConfig(
        pos_weight_type='multiplied',
        pos_weight_value=1.0,
    )

    # Specify dataset category to train on (change as needed)
    # Options: 'poctep', 'hurh', 'meg', 'eeg', 'all'
    training_category = "meg"  # Change this to switch training dataset

    # Configure dataset - should match the training category
    # For single datasets, use just that dataset name
    # For 'eeg', use ['poctep', 'hurh']
    # For 'all', use ['poctep', 'hurh', 'meg']
    if training_category in ["poctep", "hurh", "meg"]:
        dataset_names = [training_category]
    elif training_category == "eeg":
        dataset_names = ["poctep", "hurh"]
    elif training_category == "all":
        dataset_names = ["poctep", "hurh", "meg"]
    else:
        raise ValueError(f"Unknown training category: {training_category}")

    dataset_config = MultiDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        dataset_names=dataset_names,
        raw_normalization="sample-channel",
    )

    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        dataset_config=dataset_config,
        random_seed=int(os.getenv("RANDOM_SEED", 42)),
        batch_size=128,
        max_epochs=50,
        patience=10,
        min_delta=0.001,
        early_stopping_metric='loss',
        log_to_wandb=True,
        wandb_init={
            "project": "HC_vs_MCI_vs_AD_final_eval",
            "run_name": f"train_on_{training_category}_multi",
        },
        tri_class_it=True,
    )

    # Initialize logger
    magic_logger = make_logger(
        wandb_enabled=run_config.log_to_wandb,
        wandb_init=run_config.wandb_init
    )

    # Log configuration
    magic_logger.log_params(run_config.model_dump(mode='python'))
    magic_logger.log_params({"training_category": training_category})

    # Run training and evaluation
    train_and_evaluate(
        training_category=training_category,
        run_config=run_config,
        magic_logger=magic_logger,
        val_split=0.15,
    )
    
    # Finish logging
    magic_logger.finish()
    logger.success("All done!")


if __name__ == "__main__":
    main()

