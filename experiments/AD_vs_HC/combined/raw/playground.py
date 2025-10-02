#!/usr/bin/env python3
"""
Cross-validated model playground for AD_vs_HC/combined/raw experiments.

This script runs stratified cross-validation on multiple model configurations
and saves detailed results including:
- Subjects trained/validated on for each fold
- Metrics (min, max, mean, std) across folds

Modify the configs_to_test list to choose which configurations to evaluate.
"""

from core.model_playground import run_model_playground

# Define which configs to test - modify this list as needed
configs_to_test = [
    "configs/fire_lammini_style.yaml",
    "configs/nice_for_layer.yaml",
    "configs/supergood.yaml"
]

if __name__ == "__main__":
    run_model_playground(
        config_filenames=configs_to_test,
        n_folds=5,
        output_dir="playground_results"
    )
