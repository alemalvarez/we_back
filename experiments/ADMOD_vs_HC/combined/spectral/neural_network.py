from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
import torch
import numpy as np
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import time

from experiments.ADMOD_vs_HC.combined.spectral.spectral_dataset import SpectralDataset

# Configuration constants
WANDB_PROJECT = "ADMOD_vs_HC"
WANDB_CONFIG = {
    "random_seed": 42,
    "model_name": "SpectralNet_2layers",
    "input_size": 16,
    "hidden_1_size": 16,
    "hidden_2_size": 16,
    "dropout_rate": 0.24,
    "learning_rate": 0.003,
    "weight_decay": 0.0002,
    "batch_size": 32,
    "max_epochs": 50,
    "patience": 5,
    "min_delta": 0.001,
    "scaler_type": "standard",
}

ENABLE_WANDB: bool = False  # Set to False to disable wandb logging

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class SpectralNet(nn.Module):
    def __init__(self,
        input_size: int = 16,
        hidden_1_size: int = 32,
        hidden_2_size: int = 16,
        dropout_rate: float = 0.5,
    ):
        super(SpectralNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_1_size, hidden_2_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_2_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x

# Set up random seeds and device
torch.manual_seed(WANDB_CONFIG["random_seed"])
np.random.seed(WANDB_CONFIG["random_seed"]) # type: ignore

device = get_device()
logger.info(f"Using device: {device}")

now = datetime.now()

if ENABLE_WANDB:
    run = wandb.init(
        project=WANDB_PROJECT, 
        config=WANDB_CONFIG,
        name=f"{WANDB_CONFIG['model_name']}_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}"
    )
else:
    run = None

training_dataset = SpectralDataset(
    h5_file_path="artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5", 
    subjects_txt_path="experiments/ADMOD_vs_HC/combined/spectral/splits/training_subjects.txt",
    normalize=WANDB_CONFIG["scaler_type"]
)

validation_dataset = SpectralDataset(
    h5_file_path="artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5", 
    subjects_txt_path="experiments/ADMOD_vs_HC/combined/spectral/splits/validation_subjects.txt",
    normalize=WANDB_CONFIG["scaler_type"]
)

logger.info(f"Training dataset size: {len(training_dataset)}")
logger.info(f"Validation dataset size: {len(validation_dataset)}")
logger.info(f"Training subjects: {len(set(training_dataset.sample_to_subject))}")
logger.info(f"Validation subjects: {len(set(validation_dataset.sample_to_subject))}")

if ENABLE_WANDB:
    wandb.log({
        "training_dataset_size": len(training_dataset),
        "validation_dataset_size": len(validation_dataset),
    })

train_loader = DataLoader(training_dataset, batch_size=WANDB_CONFIG["batch_size"], shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=WANDB_CONFIG["batch_size"], shuffle=True)
validation_loader_no_shuffle = DataLoader(validation_dataset, batch_size=WANDB_CONFIG["batch_size"], shuffle=False)

logger.success("Data ready")

model = SpectralNet(
    input_size=WANDB_CONFIG["input_size"],
    hidden_1_size=WANDB_CONFIG["hidden_1_size"],
    hidden_2_size=WANDB_CONFIG["hidden_2_size"],
    dropout_rate=WANDB_CONFIG["dropout_rate"]
).to(device)

criterio = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr=WANDB_CONFIG["learning_rate"],
    weight_decay=WANDB_CONFIG["weight_decay"]
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if ENABLE_WANDB:
    wandb.log({
        "model_parameters": total_params,
        "trainable_parameters": trainable_params,
        "criterio": criterio,
        "optimizer": optimizer,
    })

logger.success("Model ready")

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

train_losses: list[float] = []
val_losses: list[float] = []
train_accuracies: list[float] = []
val_accuracies: list[float] = []

for epoch in range(WANDB_CONFIG["max_epochs"]):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    epoch_start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move batch to device
        data, target = data.to(device), target.to(device)
        outputs = model(data).squeeze()
        loss = criterio(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # For binary classification with BCEWithLogitsLoss
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
            data, target = data.to(device), target.to(device)
            outputs = model(data).squeeze()
            loss = criterio(outputs, target)
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
    logger.info(f"Epoch {epoch+1}/{WANDB_CONFIG['max_epochs']} completed in {train_time:.4f} seconds")
    logger.info(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}")
    logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.2f}")

    if ENABLE_WANDB:
        wandb.log({
        "train/epoch": epoch+1,
        "train/loss": avg_loss,
        "train/accuracy": avg_accuracy,
        "val/loss": avg_val_loss,
        "val/accuracy": avg_val_accuracy,
        "train/epoch_time": train_time,
        })

    if avg_val_loss < best_val_loss - WANDB_CONFIG["min_delta"]:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        if ENABLE_WANDB:
            wandb.log({"val/best_loss": avg_val_loss})
    else:
        patience_counter += 1

    if patience_counter >= WANDB_CONFIG["patience"]:
        logger.info(f"Early stopping triggered at epoch {epoch+1}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    logger.info("Best model loaded")

logger.success("Training completed")

if ENABLE_WANDB:
    wandb.log({
        "epochs": epoch+1,
        "train/loss": train_losses,
        "train/accuracy": train_accuracies,
    })

model.eval()
y_pred_list = []
y_pred_proba_list = []
y_true_list = []

# validation_loader_no_shuffle already created above

with torch.no_grad():
    for batch_X, batch_y in validation_loader_no_shuffle:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_X).squeeze()

        # Collect predictions and true labels in same order
        y_pred_proba_list.extend(outputs.cpu().numpy())
        predictions = (outputs > 0.5).float()
        y_pred_list.extend(predictions.cpu().numpy())
        y_true_list.extend(batch_y.cpu().numpy())

# Convert to numpy arrays
y_pred = np.array(y_pred_list)
y_pred_proba = np.array(y_pred_proba_list)
y_true = np.array(y_true_list)

logger.info(f"Collected {len(y_true)} predictions from validation set")
logger.info(f"Unique subjects in validation dataset: {len(set(validation_dataset.sample_to_subject))}")

# Check for potential duplicates
unique_pred_true_pairs = set(zip(y_pred, y_true))
logger.info(f"Unique (prediction, true_label) pairs: {len(unique_pred_true_pairs)}")

logger.debug(f"First 10 predictions: {y_pred[:10]}")
logger.debug(f"First 10 true labels: {y_true[:10]}")

# Create full probability matrix for compatibility with wandb plotting
y_pred_proba_full = np.column_stack([1 - y_pred_proba, y_pred_proba])

if ENABLE_WANDB:
    wandb.log({
        "val/final_accuracy": accuracy_score(y_true, y_pred),
        "val/final_f1": f1_score(y_true, y_pred),
        "val/final_precision": precision_score(y_true, y_pred),
        "val/final_recall": recall_score(y_true, y_pred),
        "val/final_roc_auc": roc_auc_score(y_true, y_pred_proba),
    })

logger.success(f"Final validation metrics:")
logger.success(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
logger.success(f"  F1-Score: {f1_score(y_true, y_pred):.4f}")
logger.success(f"  Precision: {precision_score(y_true, y_pred):.4f}")
logger.success(f"  Recall: {recall_score(y_true, y_pred):.4f}")
logger.success(f"  ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")

missclassified_indexes = np.where(y_pred != y_true)[0]
logger.warning(f"Total misclassified samples: {len(missclassified_indexes)} out of {len(y_true)} total samples")

# Only log first 10 misclassified samples to avoid spam
for i, idx in enumerate(missclassified_indexes):
    logger.warning(f"Missclassified sample {i+1}/{len(missclassified_indexes)} - Index {idx}: {validation_dataset.get_sample_to_subject(idx)}. Was classified as {y_pred[idx]} but should have been {y_true[idx]}")
