from collections.abc import Sized
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from core.schemas import BaseModelConfig
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve # type: ignore

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

def train_model(
    model: nn.Module,
    config: BaseModelConfig,
    training_dataset: Dataset,
    validation_dataset: Dataset,
) -> nn.Module:
    device = _get_device()
    model.to(device)
    logger.success("Model ready")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed) # type: ignore

    assert isinstance(training_dataset, Sized), "Training dataset must be a Sized object"
    assert isinstance(validation_dataset, Sized), "Validation dataset must be a Sized object"
    
    num_train_samples = len(training_dataset)
    num_val_samples = len(validation_dataset)
    num_train_batches = (num_train_samples + config.batch_size - 1) // config.batch_size
    num_val_batches = (num_val_samples + config.batch_size - 1) // config.batch_size

    train_pos, train_neg = _count_pos_neg(training_dataset)
    val_pos, val_neg = _count_pos_neg(validation_dataset)

    logger.info(f"Training samples: {num_train_samples}, batches: {num_train_batches}, positives: {train_pos}, negatives: {train_neg}")
    logger.info(f"Validation samples: {num_val_samples}, batches: {num_val_batches}, positives: {val_pos}, negatives: {val_neg}")

    train_loader = DataLoader(
        training_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    logger.success("Loader ready")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    logger.success("Optimizer ready")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f"Total parameters: {total_params}")
    logger.debug(f"Trainable parameters: {trainable_params}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.max_epochs):
        results = _one_epoch(
            model, 
            criterion, 
            optimizer, 
            train_loader, 
            validation_loader, 
            device,
            config,
        )

        if results.train_time > 60:
            logger.warning(f"Epoch {epoch+1} took {results.train_time:.2f} seconds, which is longer than a minute. Stopping training.")
            break

        logger.info(f"Epoch {epoch+1}/{config.max_epochs} completed in {results.train_time:.4f} seconds")
        logger.info(f"Train Loss: {results.train_loss:.4f}, Train Accuracy: {results.train_accuracy:.2f}")
        logger.info(f"Validation Loss: {results.val_loss:.4f}, Validation Accuracy: {results.val_accuracy:.2f}")
        logger.info("="*100)

        if results.val_loss < best_val_loss - config.min_delta:
            best_val_loss = results.val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.success(f"New best validation loss: {results.val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.success("Best model loaded")

    logger.success("Training completed")

    _final_evaluation(model, validation_loader, device, validation_dataset)

    return model
        
def _final_evaluation(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    validation_dataset: Dataset,
) -> None:
    model.eval()
    # y_pred_list = []
    y_pred_proba_list = []
    y_true_list = []
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device).float()
            outputs = model(data).squeeze(1)
            # predictions = (outputs > 0.5).float()
            y_pred_proba_list.extend(outputs.cpu().numpy())
            # y_pred_list.extend(predictions.cpu().numpy())
            y_true_list.extend(target.cpu().numpy())

    # y_pred = np.array(y_pred_list)
    y_pred_proba = np.array(y_pred_proba_list)
    y_true = np.array(y_true_list)
    logger.info(f"Collected {len(y_true)} predictions from validation set")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    logger.info(f"Best threshold: {best_threshold:.4f}")
    y_pred = (y_pred_proba > best_threshold).astype(int)

    assert hasattr(validation_dataset, "sample_to_subject"), "Validation dataset must have a sample_to_subject attribute"
    logger.info(f"Unique subjects in validation dataset: {len(set(validation_dataset.sample_to_subject))}")
    unique_pred_true_pairs = set(zip(y_pred, y_true))
    logger.info(f"Unique predictions and true labels pairs: {len(unique_pred_true_pairs)}")
    final_accuracy = accuracy_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_roc_auc = roc_auc_score(y_true, y_pred_proba)
    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    logger.info(f"Final F1 score: {final_f1:.4f}")
    logger.info(f"Final precision: {final_precision:.4f}")
    logger.info(f"Final recall: {final_recall:.4f}")
    logger.info(f"Final ROC AUC: {final_roc_auc:.4f}")

@dataclass
class OneEpochResults:
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    train_time: float


def _one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: BaseModelConfig,
) -> OneEpochResults:

    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    epoch_start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float()
        outputs = model(data).squeeze(1)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predictions = (outputs > 0.5).float()
        epoch_correct += (predictions == target).sum().item()
        epoch_total += target.size(0)

    avg_loss = epoch_loss / len(train_loader)
    avg_accuracy = epoch_correct / epoch_total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device).float()
            outputs = model(data).squeeze(1)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == target).sum().item()
            val_total += target.size(0)

    avg_val_loss = val_loss / len(validation_loader) # divide by number of batches my bad (its already avgd)
    avg_val_accuracy = val_correct / val_total

    train_time = time.time() - epoch_start_time

    return OneEpochResults(
        train_loss=avg_loss,
        train_accuracy=avg_accuracy,
        val_loss=avg_val_loss,
        val_accuracy=avg_val_accuracy,
        train_time=train_time,
    )