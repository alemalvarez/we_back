from datetime import datetime
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import ( # type: ignore
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    matthews_corrcoef,
)
import wandb
from dotenv import load_dotenv
from collections.abc import Sized
load_dotenv()

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

def run_sweep(model: nn.Module, run: wandb.Run, training_dataset: Dataset, validation_dataset: Dataset):

    config = run.config

    now = datetime.now()

    torch.manual_seed(42)
    np.random.seed(42)
    device = _get_device()

    run.name = f"{model.__class__.__name__}_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}"

    assert isinstance(training_dataset, Sized), "Training dataset must be a Sized object"
    assert isinstance(validation_dataset, Sized), "Validation dataset must be a Sized object"

    wandb.log({
        "training_dataset_size": len(training_dataset),
        "validation_dataset_size": len(validation_dataset),
    })

    train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    logger.success("Loader ready")
    
    model = model.to(device)

    logger.success("Model ready")

    if hasattr(config, "pos_weight") and config.pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight))
    else:
        criterion = nn.BCEWithLogitsLoss()

    if hasattr(config, "weight_decay") and config.weight_decay is not None:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:   
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    logger.success("Optimizer ready")

    wandb.watch(model, criterion, log="all", log_freq=3)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({
        "model_parameters": total_params,
        "trainable_parameters": trainable_params,
    })
    logger.debug(f"Total parameters: {total_params}")
    logger.debug(f"Trainable parameters: {trainable_params}")

    if not hasattr(config, "early_stopping_metric") or config.early_stopping_metric is None:
        config.early_stopping_metric = 'loss'  # type: ignore

    logger.info(f"Early stopping metric: {config.early_stopping_metric}")
    best_val_metric = float('inf') if config.early_stopping_metric == 'loss' else -float('inf')

    patience_counter = 0
    best_model_state = None

    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            data, target = data.to(device), target.to(device).float()
            outputs = model(data).squeeze(1)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_duration = time.time() - batch_start_time
            if batch_duration > 30:
                logger.warning(f"Batch {batch_idx+1} took {batch_duration:.2f} seconds (>30s). Aborting run.")
                return

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

        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device).float()
                logits = model(data).squeeze(1)
                loss = criterion(logits, target)
                val_loss += loss.item()

                outputs = torch.sigmoid(logits)
                predictions = (outputs > 0.5).float()

                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend(target.cpu().numpy())

                val_correct += (predictions == target).sum().item()
                val_total += target.size(0)

        avg_val_loss = val_loss / len(validation_loader)
        avg_val_accuracy = val_correct / val_total

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        train_time = time.time() - epoch_start_time
        if train_time > 60:
            logger.warning(f"Epoch {epoch+1} took {train_time:.2f} seconds, which is longer than a minute. Stopping training.")
            break

        logger.info(f"Epoch {epoch+1}/30 completed in {train_time:.4f} seconds")
        logger.info(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.2f}")

        wandb.log({
            "train/epoch": epoch+1,
            "train/loss": avg_loss,
            "train/accuracy": avg_accuracy,
            "val/loss": avg_val_loss,
            "val/accuracy": avg_val_accuracy,
            "train/epoch_time": train_time,
        })

        if config.early_stopping_metric == 'loss':
            early_stopping_metric = avg_val_loss
        elif config.early_stopping_metric == 'f1':
            early_stopping_metric = f1_score(y_true, y_pred)
        elif config.early_stopping_metric == 'mcc':
            early_stopping_metric = matthews_corrcoef(y_true, y_pred)

        if config.early_stopping_metric == 'loss':
            is_better = early_stopping_metric < best_val_metric - config.min_delta
        else:
            is_better = early_stopping_metric > best_val_metric + config.min_delta

        if is_better:
            best_val_metric = early_stopping_metric
            logger.success(f"New best validation {config.early_stopping_metric}: {early_stopping_metric:.4f}")
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(100*"=")

        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Best model loaded")

    logger.success("Training completed")

    model.eval()
    y_pred_proba_list = []
    y_true_list = []
    val_loss = 0.0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device).float()
            logits = model(data).squeeze(1)
            outputs = torch.sigmoid(logits)
            loss = criterion(logits, target)
            val_loss += loss.item()

            y_pred_proba_list.extend(outputs.cpu().numpy())
            y_true_list.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(validation_loader)

    wandb.log({
        "val/loss": avg_val_loss,
    })

    logger.info(f"Validation Loss: {avg_val_loss:.4f}")

    y_pred_proba = np.array(y_pred_proba_list)
    y_true = np.array(y_true_list)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    logger.info(f"Best threshold: {best_threshold:.4f}")

    y_pred = (y_pred_proba > best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    logger.info(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    wandb.log({
        "val/tn": tn,
        "val/fp": fp,
        "val/fn": fn,
        "val/tp": tp,
    })

    final_accuracy = accuracy_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_roc_auc = roc_auc_score(y_true, y_pred_proba)
    final_mcc = matthews_corrcoef(y_true, y_pred)

    wandb.log({
        "val/final_accuracy": final_accuracy,
        "val/final_f1": final_f1,
        "val/final_precision": final_precision,
        "val/final_recall": final_recall,
        "val/final_roc_auc": final_roc_auc,
        "val/final_mcc": final_mcc,
    })

    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    logger.info(f"Final F1 score: {final_f1:.4f}")
    logger.info(f"Final precision: {final_precision:.4f}")
    logger.info(f"Final recall: {final_recall:.4f}")
    logger.info(f"Final ROC AUC: {final_roc_auc:.4f}")
    logger.info(f"Final MCC: {final_mcc:.4f}")
        
        