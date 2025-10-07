from core.builders import build_dataset
from core.schemas import OptimizerConfig, CriterionConfig, RawDatasetConfig, RunConfig
from dotenv import load_dotenv
import os
import sys
from loguru import logger
import yaml # type: ignore[import]
from models.simple_2d import NETWORK_CONFIG_CLASSES
from core.sanity_test_model import sanity_test_model
from core.schemas import filter_config_params

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Get config file path from command line argument or use default
if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "custom.yaml"
    logger.info(f"No config file specified, using default: {config_file}")

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
config_dict["pos_weight_type"] = "fixed"
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

sanity_test_model(
    run_config,
    training_dataset, 
    validation_dataset,
    run_overfit_test=True, 
    overfit_test_epochs=100
)