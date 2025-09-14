from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score # type: ignore
from torch import nn
import torch
import numpy as np
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import time
import argparse

import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(__file__))

from core.spectral_dataset import SpectralDataset

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


def main():
    parser = argparse.ArgumentParser(description='Train SpectralNet with wandb sweep')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_1_size', type=int, default=32)
    parser.add_argument('--hidden_2_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    # Set up random seeds and device
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize wandb
    wandb.init(project="ADSEV_vs_HC_sweep", config=vars(args))
    config = wandb.config

    now = datetime.now()
    wandb.run.name = f"SpectralNet_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}"

    # Create datasets
    training_dataset = SpectralDataset(
        h5_file_path="artifacts/POCTEP_DK_features_only:v0/POCTEP_DK_features_only.h5",
        subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/spectral/splits/training_subjects.txt",
        normalize="standard"
    )

    validation_dataset = SpectralDataset(
        h5_file_path="artifacts/POCTEP_DK_features_only:v0/POCTEP_DK_features_only.h5",
        subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/spectral/splits/validation_subjects.txt",
        normalize="standard"
    )

    wandb.log({
        "training_dataset_size": len(training_dataset),
        "validation_dataset_size": len(validation_dataset),
    })

    train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader_no_shuffle = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

    logger.info("Data ready")

    # Create model
    model = SpectralNet(
        input_size=16,
        hidden_1_size=config.hidden_1_size,
        hidden_2_size=config.hidden_2_size,
        dropout_rate=config.dropout_rate
    ).to(device)

    criterio = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    wandb.log({
        "model_parameters": total_params,
        "trainable_parameters": trainable_params,
    })

    logger.info("Model ready")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data).squeeze()
            loss = criterio(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = (outputs > 0.5).float()
            epoch_correct += (predictions == target).sum().item()
            epoch_total += target.size(0)

        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_correct / epoch_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in validation_loader_no_shuffle:
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
        logger.info(f"Epoch {epoch+1}/{config.max_epochs} completed in {train_time:.4f} seconds")
        logger.info(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}%")
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.2f}%")

        wandb.log({
            "train/epoch": epoch+1,
            "train/loss": avg_loss,
            "train/accuracy": avg_accuracy,
            "val/loss": avg_val_loss,
            "val/accuracy": avg_val_accuracy,
            "train/epoch_time": train_time,
        })

        # Early stopping check
        if avg_val_loss < best_val_loss - 0.001:  # min_delta
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            wandb.log({"val/best_loss": avg_val_loss})
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Best model loaded")

    logger.info("Training completed")

    # Final evaluation
    model.eval()
    y_pred_list = []
    y_pred_proba_list = []
    y_true_list = []

    with torch.no_grad():
        for batch_X, batch_y in validation_loader_no_shuffle:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X).squeeze()

            y_pred_proba_list.extend(outputs.cpu().numpy())
            predictions = (outputs > 0.5).float()
            y_pred_list.extend(predictions.cpu().numpy())
            y_true_list.extend(batch_y.cpu().numpy())

    y_pred = np.array(y_pred_list)
    y_pred_proba = np.array(y_pred_proba_list)
    y_true = np.array(y_true_list)

    # Log final metrics
    final_accuracy = accuracy_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_roc_auc = roc_auc_score(y_true, y_pred_proba)

    wandb.log({
        "val/final_accuracy": final_accuracy,
        "val/final_f1": final_f1,
        "val/final_precision": final_precision,
        "val/final_recall": final_recall,
        "val/final_roc_auc": final_roc_auc,
        "epochs": epoch+1,
    })

    logger.info("Final validation metrics:")
    logger.info(f"  Accuracy: {final_accuracy:.4f}")
    logger.info(f"  F1-Score: {final_f1:.4f}")
    logger.info(f"  Precision: {final_precision:.4f}")
    logger.info(f"  Recall: {final_recall:.4f}")
    logger.info(f"  ROC-AUC: {final_roc_auc:.4f}")


if __name__ == "__main__":
    main()
