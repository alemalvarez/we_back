from core.builders import build_dataset
from core.schemas import OptimizerConfig, CriterionConfig, RawDatasetConfig, RunConfig
from dotenv import load_dotenv
import os
import sys
from loguru import logger
import torch
import yaml
from models.simple_2d import DeeperCustomConfig
from core.runner import run as run_single
from core.logging import make_logger

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")

# Get config file path from command line argument
if len(sys.argv) != 2:
    logger.error("Usage: python train.py <config_file>")
    logger.error("Example: python train.py configs/Improved2D_0928_2024.yaml")
    sys.exit(1)

config_file = sys.argv[1]
logger.info(f"Using config file: {config_file}")

# Load configuration using the universal loader
if os.path.isabs(config_file):
    config_path = config_file
else:
    config_path = f"experiments/AD_vs_HC/combined/raw/{config_file}"
with open(config_path) as f:
    config_dict = yaml.safe_load(f)

network_config = DeeperCustomConfig(
    model_name="DeeperCustom",
    n_filters=config_dict["n_filters"],
    kernel_sizes=config_dict["kernel_sizes"],
    strides=config_dict["strides"],
    dropout_rate=config_dict["dropout_rate"],
    paddings=config_dict["paddings"],
    activation=config_dict["activation"],
)

optimizer_config = OptimizerConfig(
    learning_rate=config_dict["learning_rate"],
    weight_decay=config_dict["weight_decay"],
    use_cosine_annealing=config_dict["use_cosine_annealing"],
    cosine_annealing_t_0=config_dict["cosine_annealing_t_0"],
    cosine_annealing_t_mult=config_dict["cosine_annealing_t_mult"],
    cosine_annealing_eta_min=config_dict["cosine_annealing_eta_min"],
)

criterion_config = CriterionConfig(
    pos_weight_type=config_dict["pos_weight_type"],
    pos_weight_value=config_dict["pos_weight_value"],
)
dataset_config = RawDatasetConfig(
    h5_file_path=H5_FILE_PATH,
    dataset_type="raw",
    raw_normalization=config_dict["normalize"],
    augment=config_dict["augment"],
    augment_prob=(config_dict["augment_prob_neg"], config_dict["augment_prob_pos"]),
    noise_std=config_dict["noise_std"],
)

run_config = RunConfig(
    network_config=network_config,
    optimizer_config=optimizer_config,
    criterion_config=criterion_config,
    dataset_config=dataset_config,
    random_seed=config_dict["random_seed"],
    batch_size=config_dict["batch_size"],
    max_epochs=config_dict["max_epochs"],
    patience=config_dict["patience"],
    min_delta=config_dict["min_delta"],
    early_stopping_metric=config_dict["early_stopping_metric"],
    log_to_wandb=False,
    wandb_init=None,
)

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