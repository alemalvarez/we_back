from typing import Optional
from core.schemas import RunConfig
from core.builders import build_model, build_optimizer, build_scheduler, build_criterion
from core.logging import make_logger, Logger
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from loguru import logger

from core.training_loop import one_epoch, OneEpochResults

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.success("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.success("Using MPS")
        return torch.device("mps")
    else:
        logger.info("Using CPU...")
        return torch.device("cpu")  

def _count_pos_neg(dataset: Dataset) -> tuple[int, int]:
    # Assumes dataset[i][1] is the label and is 0 or 1 or convertible to int
    pos = 0
    neg = 0
    for i in range(len(dataset)): # type: ignore
        label = dataset[i][1]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if int(label) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg

def run(
    config: RunConfig,
    training_dataset: Dataset,
    validation_dataset: Dataset,
    logger_sink: Optional[Logger] = None,
    ):
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    device = _get_device()

    train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    model = build_model(config.network_config)
    model = model.to(device)

    optimizer = build_optimizer(config.optimizer_config, model)
    scheduler = build_scheduler(config.optimizer_config, optimizer)
    criterion = build_criterion(config.criterion_config, _count_pos_neg(training_dataset))
    magic_logger = logger_sink or make_logger(wandb_enabled=config.log_to_wandb, wandb_init=config.wandb_init)
    magic_logger.watch(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    magic_logger.log_metrics({
        "model/parameters_total": float(total_params),
        "model/parameters_trainable": float(trainable_params),
    }, step=0)

    if config.early_stopping_metric == 'loss':
        best_metric = float('inf')
        def better(current: float, best: float) -> bool:
            return current < best - config.min_delta
    else:
        best_metric = -float('inf')
        def better(current: float, best: float) -> bool:
            return current > best + config.min_delta

    best_state = None
    patience_counter = 0

    for epoch in range(config.max_epochs):
        results: OneEpochResults = one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            validation_loader,
            device,
        )

        logger.info(f"Epoch {epoch+1}/{config.max_epochs} | "
                    f"train_loss={results.train_loss:.4f} acc={results.train_accuracy:.2f} | "
                    f"val_loss={results.val_loss:.4f} acc={results.val_accuracy:.2f} "
                    f"f1={results.val_f1:.4f} mcc={results.val_mcc:.4f} kappa={results.val_kappa:.4f} | "
                    f"time={results.epoch_time:.2f}s")

        magic_logger.log_metrics({
            "train/loss": results.train_loss,
            "train/accuracy": results.train_accuracy,
            "val/loss": results.val_loss,
            "val/accuracy": results.val_accuracy,
            "val/f1": results.val_f1,
            "val/mcc": results.val_mcc,
            "val/kappa": results.val_kappa,
            "epoch/time": results.epoch_time,
        }, step=epoch)

        if config.early_stopping_metric == 'loss':
            current_metric = results.val_loss
        elif config.early_stopping_metric == 'f1':
            current_metric = results.val_f1
        elif config.early_stopping_metric == 'mcc':
            current_metric = results.val_mcc
        elif config.early_stopping_metric == 'kappa':
            current_metric = results.val_kappa
        else:
            current_metric = results.val_loss

        if better(current_metric, best_metric):
            best_metric = current_metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.success(f"New best {config.early_stopping_metric}: {best_metric:.4f}")
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()
            logger.info(f"Learning rate after annealing: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)  # type: ignore[arg-type]
        logger.success("Best model weights restored")

    if logger_sink is None:
        magic_logger.finish()

    return model