from core.builders import build_dataset
from core.schemas import OptimizerConfig, CriterionConfig, RawDatasetConfig, RunConfig
from dotenv import load_dotenv
import os
from loguru import logger
import torch
from core.runner import run as run_single
from core.logging import make_logger
from models.squeezer import DeeperSEConfig

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

model_config = DeeperSEConfig(
    model_name="DeeperSE",
    n_filters=[16, 32, 64, 128],
    kernel_sizes=[(100, 3), (15, 10), (10, 3), (5, 2)],
    strides=[(2, 2), (2, 2), (1, 1), (1, 1)],
    dropout_rate=0.31158910319253397,
    paddings=[(25, 1), (5, 2), (5, 1), (1, 1)],
    activation="silu",
    reduction_ratio=4,
)
optimizer_config = OptimizerConfig(
    learning_rate=0.004255107493153422,
    weight_decay=9.6832252733516e-05,
    use_cosine_annealing=False,
    cosine_annealing_t_0=8,
    cosine_annealing_t_mult=2,
    cosine_annealing_eta_min=1e-6,
)
criterion_config = CriterionConfig(
    pos_weight_type='multiplied',
    pos_weight_value=1.09784373282656,
)

dataset_config = RawDatasetConfig(
    h5_file_path=H5_FILE_PATH,
    raw_normalization='channel-subject',
    augment=False,
)

run_config = RunConfig(
    network_config=model_config,
    optimizer_config=optimizer_config,
    criterion_config=criterion_config,
    random_seed=42,
    batch_size=128,
    max_epochs=50,
    patience=5,
    min_delta=0.001,
    early_stopping_metric='loss',
    dataset_config=dataset_config,
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
save_path = os.path.join(script_dir, f"{model_config.model_name}_trained.pt")
torch.save(trained_model.state_dict(), save_path)
logger.success(f"Saved trained model to {save_path}")