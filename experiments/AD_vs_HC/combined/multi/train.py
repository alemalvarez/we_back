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

from models.shallow_concatter import ShallowerConcatterConfig


load_dotenv()


H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")


if __name__ == "__main__":
    # To address overfitting, you can try several strategies:
    # 1. Increase dropout rates.
    # 2. Use stronger regularization, e.g., increase weight_decay in optimizer.
    # 3. Try reducing model complexity (fewer filters, smaller hidden sizes).
    # 4. Add early stopping or make it more aggressive (lower patience/min_delta).
    # 5. Increase dataset size or augment data if possible.

    # BREAKTHROUGH CONFIG: Keep aggressive downsampling, slightly more capacity
    model_config = ShallowerConcatterConfig(
        model_name="ShallowerConcatter",
        n_filters=[8, 16],  # gentle increase from 4,8
        kernel_sizes=[(50, 2), (10, 5)],  # KEEP these - they work
        strides=[(10, 5), (8, 4)],  # KEEP aggressive downsampling - this is KEY
        raw_dropout_rate=0.70,  # reduce slightly from 0.75
        paddings=[(5, 0), (1, 1)],
        activation="silu",
        n_spectral_features=16,
        spectral_hidden_size=64,
        spectral_dropout_rate=0.25,
        concat_dropout_rate=0.50,  # reduce from 0.55
        fusion_hidden_size=128,
    )

    optimizer_config = OptimizerConfig(
        learning_rate=0.006,  # slightly higher initial LR
        weight_decay=0.003,  # reduce L2 a bit to allow more learning
        use_cosine_annealing=True,
        cosine_annealing_t_0=10,
        cosine_annealing_t_mult=2,
        cosine_annealing_eta_min=1e-5,
    )

    criterion_config = CriterionConfig(
        pos_weight_type='multiplied',
        pos_weight_value=1.0,
    )

    dataset_config = MultiDatasetConfig(
        h5_file_path=H5_FILE_PATH,
        raw_normalization='sample-channel',  # per-sample normalization
        spectral_normalization='standard',
    )

    run_config = RunConfig(
        network_config=model_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=42,
        batch_size=64,  # smaller batches = more gradient noise = regularization
        max_epochs=100,  # allow longer training like spectral
        patience=15,  # more patience
        min_delta=0.0005,  # less aggressive stopping
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