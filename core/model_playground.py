import os
import yaml # type: ignore[import]
import json
from pathlib import Path
from typing import List, Tuple, Literal
from dataclasses import dataclass
from dotenv import load_dotenv

from core.run_cv import run_cv
from models.simple_2d import DeeperCustom, Deeper2D, Improved2D, Simple2D3Layers
from loguru import logger
from core.schemas import BaseModelConfig
import torch.nn as nn

load_dotenv()

# Config classes for different model types
@dataclass
class CustomConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float
    paddings: List[Tuple[int, int]]
    activation: str
    augment: bool
    augment_prob_pos: float
    augment_prob_neg: float
    noise_std: float

@dataclass
class Deeper2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float
    paddings: List[Tuple[int, int]]

@dataclass
class Improved2DConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float
    paddings: List[Tuple[int, int]]

@dataclass
class Simple2D3LayersConfig(BaseModelConfig):
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    pos_weight: float

def create_model(config: BaseModelConfig) -> nn.Module:
    """Create model instance based on config."""
    if isinstance(config, CustomConfig):
        return DeeperCustom(
            n_filters=config.n_filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            dropout_rate=config.dropout_rate,
            paddings=config.paddings,
            activation=config.activation
        )
    elif isinstance(config, Deeper2DConfig):
        return Deeper2D(
            n_filters=config.n_filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            dropout_rate=config.dropout_rate,
            paddings=config.paddings
        )
    elif isinstance(config, Improved2DConfig):
        return Improved2D(
            n_filters=config.n_filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            dropout_rate=config.dropout_rate,
            paddings=config.paddings
        )
    elif isinstance(config, Simple2D3LayersConfig):
        return Simple2D3Layers(
            n_filters=config.n_filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            dropout_rate=config.dropout_rate
        )
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

def load_subjects() -> List[str]:
    """Load subjects from training and validation split files."""
    subjects = []
    for path in [
        "experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
        "experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
    ]:
        with open(path, "r") as f:
            subjects.extend([line.strip() for line in f if line.strip()])
    return subjects

def load_config(config_filename: str) -> BaseModelConfig:
    """Load config from experiments/AD_vs_HC/combined/raw/ directory."""
    # Handle both absolute paths and relative filenames
    if os.path.isabs(config_filename):
        config_path = config_filename
    else:
        config_path = f"experiments/AD_vs_HC/combined/raw/{config_filename}"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config_dict["random_seed"] = int(os.getenv("RANDOM_SEED", "42"))

    # Determine config class based on model_name
    model_name = config_dict["model_name"]
    if model_name == "DeeperCustom":
        return CustomConfig(**config_dict)
    elif model_name == "Deeper2D":
        return Deeper2DConfig(**config_dict)
    elif model_name == "Improved2D":
        return Improved2DConfig(**config_dict)
    elif model_name == "Simple2D_3layers":
        return Simple2D3LayersConfig(**config_dict)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def run_model_playground(config_filenames: List[str], n_folds: int = 5, output_dir: str = "playground_results", h5_file_path: str = "h5test_raw_only.h5"):
    """
    Run cross-validation playground for multiple configs.

    Args:
        config_filenames: List of config filenames (e.g., ['config1.yaml', 'config2.yaml'])
        n_folds: Number of CV folds
        output_dir: Directory to save results
    """

    subjects = load_subjects()
    results = {}

    for config_filename in config_filenames:
        logger.info(f"Running playground for config: {config_filename}")

        # Load config and create model
        config = load_config(config_filename)
        model = create_model(config)

        # Run CV
        cv_results = run_cv(
            model=model,
            config=config,
            h5_file_path=h5_file_path,
            included_subjects=subjects,
            n_folds=n_folds
        )

        # Store results
        results[config_filename] = {
            "model_name": config.model_name,
            "folds": [
                {
                    "fold_idx": i,
                    "train_subjects": fold_detail["train_subjects"],
                    "val_subjects": fold_detail["val_subjects"],
                    "metrics": fold_metrics
                }
                for i, (fold_metrics, fold_detail) in enumerate(
                    zip(cv_results["folds"], cv_results["fold_details"])
                )
            ],
            "aggregate": cv_results["aggregate"]
        }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    with open(output_path / "playground_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Results saved to {output_path}/playground_results.json")
    return results
