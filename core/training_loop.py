from dataclasses import dataclass
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from loguru import logger
from sklearn.metrics import ( # type: ignore
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

def _unpack_batch(batch: tuple, device: torch.device) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Unpack batch and move to device. Handles both (x, y) and ((x_raw, x_spectral), y) formats."""
    data_or_tuple, target = batch  # type: ignore
    
    if isinstance(data_or_tuple, (tuple, list)):
        data: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = tuple(d.to(device) for d in data_or_tuple)
    else:
        data = data_or_tuple.to(device)
    
    target = target.to(device).float()
    return data, target

# Training config consts (shouldn't be changed often)
MAX_EPOCH_ALLOWED_TIME = 30 # seconds
MAX_AVG_BATCH_TIME = 2 # seconds
MIN_BATCHES_FOR_AVG = 10 # minimum number of batches before checking average
SHOULD_CLIP_GRADIENT = True
CLIP_GRADIENT_MAX_NORM = 1.0
COSINE_ANNEALING_T_0 = 5
COSINE_ANNEALING_T_MULT = 1
COSINE_ANNEALING_ETA_MIN = 1e-6

class UnefficientRun(Exception):
    """
    Exception raised when a run is unefficient.
    """
    pass

@dataclass
class OneEpochResults:
    train_loss: float
    train_accuracy: float
    epoch_time: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_mcc: float
    val_kappa: float

def one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> OneEpochResults:
    """
    One epoch of training, plus a validation step for it.
    Args:
        model: The model to train.
        criterion: The criterion to use.
        optimizer: The optimizer to use.
        train_loader: The train loader.
        validation_loader: The validation loader.
        device: The device to use.
    Returns:
        OneEpochResults: The results of the epoch, containing:
            - epoch_number: The number of the epoch.
            - train_loss: The loss on the train set.
            - train_accuracy: The accuracy on the train set.
            - epoch_time: The time it took to train the epoch.
            - val_loss: The loss on the validation set.
            - val_accuracy: The accuracy on the validation set.
            - val_f1: The F1 score on the validation set.
            - val_mcc: The MCC score on the validation set.
            - val_kappa: The Kappa score on the validation set.
    """
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    epoch_start_time = time.time()
    batch_durations: list[float] = []

    for batch_idx, batch in enumerate(train_loader): # TODO: it could be interesting to get metrics 
        batch_start_time = time.time()
        # on a batch level. maybe later.
        data, target = _unpack_batch(batch, device)
        logits = model(data).squeeze(1) # well, i don't usually put the sigmoid on the model.
        loss = criterion(logits, target) # im kind of expecting that this will be BCEWithLogitsLoss. i 
        # should probably check what happens if it's not.

        optimizer.zero_grad()
        loss.backward()
        if SHOULD_CLIP_GRADIENT:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRADIENT_MAX_NORM)
        optimizer.step()

        batch_duration = time.time() - batch_start_time
        batch_durations.append(batch_duration)
        
        # Check running average once we have enough batches
        if len(batch_durations) >= MIN_BATCHES_FOR_AVG:
            avg_batch_time = sum(batch_durations) / len(batch_durations)
            if avg_batch_time > MAX_AVG_BATCH_TIME:
                logger.warning(
                    f"Average batch time ({avg_batch_time:.2f}s) exceeds threshold "
                    f"({MAX_AVG_BATCH_TIME}s) after {len(batch_durations)} batches. Aborting run."
                )
                raise UnefficientRun("Average batch time exceeds threshold. Aborting run.")

        epoch_loss += loss.item()
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        epoch_correct += (predictions == target).sum().item()
        epoch_total += target.size(0)

    avg_loss = epoch_loss / len(train_loader)
    avg_accuracy = epoch_correct / epoch_total

    # ----- validate    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in validation_loader:
            data, target = _unpack_batch(batch, device)
            logits = model(data).squeeze(1)

            loss = criterion(logits, target)
            val_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float() # this is quite 
            # arbitrary, im just picking .5. 

            y_pred_list.extend(predictions.cpu().numpy())
            y_true_list.extend(target.cpu().numpy())

            val_correct += (predictions == target).sum().item()
            val_total += target.size(0)

    avg_val_loss = val_loss / len(validation_loader)
    avg_val_accuracy = val_correct / val_total

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    epoch_time = time.time() - epoch_start_time
    if epoch_time > MAX_EPOCH_ALLOWED_TIME:
        logger.warning(f"Epoch took {epoch_time:.2f} seconds (>{MAX_EPOCH_ALLOWED_TIME}s). Aborting run.")
        raise UnefficientRun("Epoch took too long to process. Aborting run.")

    val_f1 = f1_score(y_true, y_pred)
    val_mcc = matthews_corrcoef(y_true, y_pred)
    val_kappa = cohen_kappa_score(y_true, y_pred)

    # This will return a successful epoch. Early stopping will be handled one layer outside.
    return OneEpochResults(
        train_loss=avg_loss,
        train_accuracy=avg_accuracy,
        val_loss=avg_val_loss,
        val_accuracy=avg_val_accuracy,
        val_f1=val_f1,
        val_mcc=val_mcc,
        val_kappa=val_kappa,
        epoch_time=epoch_time,
    )
