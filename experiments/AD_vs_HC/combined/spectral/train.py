import os
import torch
from dotenv import load_dotenv

from core.spectral_dataset import SpectralDataset
from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
)
from core.runner import run as run_single
from core.evaluation import evaluate_with_config, pretty_print_per_subject
from core.logging import make_logger

from models.spectral_net import SpectralNetConfig


load_dotenv()


H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")


if __name__ == "__main__":
    network_config = SpectralNetConfig(
        model_name="SpectralNet",
        input_size=16,
        hidden_1_size=32,
        hidden_2_size=16,
        dropout_rate=0.5,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=0.003111076215981144,
        weight_decay=0.00027819671966625116,
        use_cosine_annealing=False,
    )
    criterion_config = CriterionConfig(
        pos_weight_type='fixed',
        pos_weight_value=1.0,
    )

    run_config = RunConfig(
        network_config=network_config,
        optimizer_config=optimizer_config,
        criterion_config=criterion_config,
        random_seed=42,
        batch_size=32,
        max_epochs=50,
        patience=5,
        min_delta=0.001,
        early_stopping_metric='mcc',
        normalization='standard',
        log_to_wandb=False,
    )

    training_dataset = SpectralDataset(
        h5_file_path=H5_FILE_PATH,
        subjects_txt_path="experiments/AD_vs_HC/combined/spectral/splits/training_subjects.txt",
        normalize=run_config.normalization,
    )

    validation_dataset = SpectralDataset(
        h5_file_path=H5_FILE_PATH,
        subjects_txt_path="experiments/AD_vs_HC/combined/spectral/splits/validation_subjects.txt",
        normalize=run_config.normalization,
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
    save_path = os.path.join(script_dir, f"{network_config.model_name}_trained.pt")
    torch.save(trained_model.state_dict(), save_path)
    
    magic_logger.finish()