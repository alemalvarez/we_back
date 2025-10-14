import os

from core.schemas import (
    OptimizerConfig,
    CriterionConfig,
    RunConfig,
    RawDatasetConfig,
)
from core.cv import build_dataset
from core.sanity_test_model import sanity_test_model

from models.squeezer import DeeperSEConfig

from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    h5_file_path = os.getenv(
        "H5_FILE_PATH",
        "h5test_raw_only.h5",
    )

    splits_dir = "experiments/AD_vs_HC/combined/raw/splits"
    train_subjects_path = os.path.join(splits_dir, "training_subjects.txt")

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
        h5_file_path=h5_file_path,
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
        subjects_path=train_subjects_path,
        validation=False
    )

    sanity_test_model(run_config, training_dataset, run_overfit_test=True, overfit_test_epochs=50)


if __name__ == "__main__":
    main()