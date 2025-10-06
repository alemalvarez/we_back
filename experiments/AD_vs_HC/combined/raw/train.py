from core.builders import build_dataset
from core.schemas import OptimizerConfig, CriterionConfig, RawDatasetConfig, RunConfig
from dotenv import load_dotenv
import os
import sys
from loguru import logger
import torch
import yaml # type: ignore[import]
from models.simple_2d import NETWORK_CONFIG_CLASSES
from core.runner import run as run_single
from core.logging import make_logger
from core.schemas import filter_config_params

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Get config file path from command line argument
if len(sys.argv) != 2:
    logger.error("Usage: python train.py <config_file>")
    logger.error("Example: python train.py configs/Improved2D_0928_2024.yaml")
    sys.exit(1)

config_file = sys.argv[1]
logger.info(f"Using config file: {config_file}")

if os.path.isabs(config_file):
    config_path = config_file
else:
    config_path = f"experiments/AD_vs_HC/combined/raw/{config_file}"
with open(config_path) as f:
    config_dict = yaml.safe_load(f)

# Handle parameter name mappings
config_dict["model_name"] = config_dict.get("model_name", "DeeperCustom")
config_dict["pos_weight_value"] = config_dict.get("pos_weight", 1.0)
config_dict["pos_weight_type"] = "fixed"  # Default to fixed, could be made configurable
config_dict["augment_prob"] = (config_dict.get("augment_prob_neg", 0.5), config_dict.get("augment_prob_pos", 0.0))

# Dynamically choose network config class based on model_name
model_name = config_dict["model_name"]
if model_name not in NETWORK_CONFIG_CLASSES:
    available_models = list(NETWORK_CONFIG_CLASSES.keys())
    logger.error(f"Unknown model_name '{model_name}'. Available models: {available_models}")
    sys.exit(1)

network_config_class = NETWORK_CONFIG_CLASSES[model_name]
network_config = network_config_class(**filter_config_params(network_config_class, config_dict)) # type: ignore[arg-type]
optimizer_config = OptimizerConfig(**filter_config_params(OptimizerConfig, config_dict))
criterion_config = CriterionConfig(**filter_config_params(CriterionConfig, config_dict))

# Dataset config needs special handling for h5_file_path
dataset_params = filter_config_params(RawDatasetConfig, config_dict)
dataset_params["h5_file_path"] = H5_FILE_PATH
dataset_params["dataset_type"] = "raw"
dataset_config = RawDatasetConfig(**dataset_params)

# Run config needs special handling for nested configs and defaults
run_params = filter_config_params(RunConfig, config_dict)
run_params.update({
    "network_config": network_config,
    "optimizer_config": optimizer_config,
    "criterion_config": criterion_config,
    "dataset_config": dataset_config,
    "log_to_wandb": False,
    "wandb_init": None,
    "random_seed": RANDOM_SEED,
})
run_config = RunConfig(**run_params)

training_dataset = build_dataset(
    dataset_config,
    subjects_path="experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
    validation=False
)

validation_dataset = build_dataset(
    dataset_config,
    subjects_path="experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
    validation=True
)

magic_logger = make_logger(wandb_enabled=run_config.log_to_wandb, wandb_init=run_config.wandb_init)

trained_model = run_single(
    config=run_config,
    training_dataset=training_dataset,
    validation_dataset=validation_dataset,
    logger_sink=magic_logger,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, f"{network_config.model_name}_trained.pt")
torch.save(trained_model.state_dict(), save_path)
logger.success(f"Saved trained model to {save_path}")