import os
import torch
from dotenv import load_dotenv

from core.builders import build_dataset
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    MultiDatasetConfig,
)
from core.runner import run as run_single
from core.evaluation import evaluate_with_config, pretty_print_per_subject
from core.logging import make_logger

from models.concatter import ConcatterConfig, GatedConcatterConfig


load_dotenv()


H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")


if __name__ == "__main__":
    model_config = GatedConcatterConfig(
        model_name="GatedConcatter",
        n_filters=[16, 32, 64, 128],
        kernel_sizes=[(100, 3), (15, 10), (10, 3), (5, 2)],
        strides=[(2, 2), (2, 2), (1, 1), (1, 1)],
        raw_dropout_rate=0.31158910319253397,
        paddings=[(25, 1), (5, 2), (5, 1), (1, 1)],
        activation="silu",
        n_spectral_features=16,
        spectral_dropout_rate=0.5,
        head_hidden_sizes=[128, 32],
        concat_dropout_rate=0.31158910319253397,
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

    dataset_config = MultiDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        raw_normalization='channel-subject',
        spectral_normalization='standard',
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
        subjects_path="experiments/AD_vs_HC/combined/multi/splits/training_subjects.txt",
        validation=False
    )

    validation_dataset = build_dataset(
        dataset_config,
        subjects_path="experiments/AD_vs_HC/combined/multi/splits/validation_subjects.txt",
        validation=True
    )

    magic_logger = make_logger(wandb_enabled=run_config.log_to_wandb, wandb_init=run_config.wandb_init)

    trained_model = run_single(
        config=run_config,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        logger_sink=magic_logger,
    )
    
    result = evaluate_with_config(
        model=trained_model,
        dataset=validation_dataset,
        run_config=run_config,
        logger_sink=magic_logger,
        prefix="val",
    )

    pretty_print_per_subject(result.per_subject)
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f"{model_config.model_name}_trained.pt")
    torch.save(trained_model.state_dict(), save_path)
    
    magic_logger.finish()