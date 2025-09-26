from dataclasses import dataclass
from typing import List, Tuple, Literal
import time

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from core.raw_dataset import RawDataset
from models.simple_2d import Simple2D3Layers

WANDB_PROJECT = "ADSEV_vs_HC"
WANDB_CONFIG = {
    "random_seed": 42,
    "model_name": "Simple2D_3layers",
    "n_filters": [16, 32, 64],
    "kernel_sizes": [(5, 5), (5, 5), (5, 5)],
    "strides": [(1, 1), (1, 1), (1, 1)],
    "dropout_rate": 0.25,
    "input_shape": (1000, 1, 68),
    "learning_rate": 0.001,
    "batch_size": 128,
    "max_epochs": 50,
    "patience": 20,
    "min_delta": 0.001,
    "normalize": "sample-channel",
}

@dataclass
class ModelConfig:
    random_seed: int
    model_name: str
    n_filters: List[int]
    kernel_sizes: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    dropout_rate: float
    input_shape: Tuple[int, int, int]
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']

config = ModelConfig(**WANDB_CONFIG) # type: ignore

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed) # type: ignore

device = get_device()
logger.info(f"Using device: {device}")

training_dataset = RawDataset(
    h5_file_path="POCTEP_raw_only.h5",
    subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/raw/splits/training_subjects.txt",
    normalize=config.normalize
)

validation_dataset = RawDataset(
    h5_file_path="POCTEP_raw_only.h5",
    subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/raw/splits/validation_subjects.txt",
    normalize=config.normalize
)

train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True)
validation_loader_no_shuffle = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

logger.success("Data ready")

model = Simple2D3Layers(
    n_filters=config.n_filters,
    kernel_sizes=config.kernel_sizes,
    strides=config.strides,
    dropout_rate=config.dropout_rate,
).to(device)

logger.success("Model ready")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

logger.success("Optimizer ready")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.success(f"Total parameters: {total_params}")
logger.success(f"Trainable parameters: {trainable_params}")

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

train_losses: list[float] = []
val_losses: list[float] = []
train_accuracies: list[float] = []
val_accuracies: list[float] = []

for epoch in range(config.max_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    epoch_start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move batch to device
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
        for data, target in validation_loader_no_shuffle:  # Use unshuffled for consistent metrics
            data, target = data.to(device), target.to(device).float()
            outputs = model(data).squeeze(1)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == target).sum().item()
            val_total += target.size(0)

    avg_val_loss = val_loss / len(validation_loader_no_shuffle)
    avg_val_accuracy = val_correct / val_total

    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    train_time = time.time() - epoch_start_time
    logger.info(f"Epoch {epoch+1}/{config.max_epochs} completed in {train_time:.4f} seconds")
    logger.info(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}%")
    logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss - config.min_delta:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        logger.success(f"New best validation loss: {avg_val_loss:.4f}")

    else:
        patience_counter += 1

    if patience_counter >= config.patience:
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    logger.success("Best model loaded")

logger.success("Training completed")

model.eval()
y_pred_list = []
y_pred_proba_list = []
y_true_list = []

with torch.no_grad():
    for data, target in validation_loader_no_shuffle:
        data, target = data.to(device), target.to(device)
        outputs = model(data).squeeze(1)
        y_pred_proba_list.extend(outputs.cpu().numpy())

        predictions = (outputs > 0.5).float()
        y_pred_list.extend(predictions.cpu().numpy())
        y_true_list.extend(target.cpu().numpy())

y_pred = np.array(y_pred_list)
y_pred_proba = np.array(y_pred_proba_list)
y_true = np.array(y_true_list)

logger.info(f"Collected {len(y_true)} predictions from validation set")
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