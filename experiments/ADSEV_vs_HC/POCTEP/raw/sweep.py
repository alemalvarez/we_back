from datetime import datetime
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from core.raw_dataset import RawDataset
from models.simple_2d import Simple2D
from loguru import logger
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
import wandb
from dotenv import load_dotenv
import os
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

WANDB_PROJECT = "ADSEV_vs_HC_sweep"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")

def main():
    with wandb.init(project=WANDB_PROJECT) as run:
        config = run.config

        now = datetime.now()

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        device = _get_device()

        wandb.run.name = f"Simple2D_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}"

        training_dataset = RawDataset(
            h5_file_path=H5_FILE_PATH,
            subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/raw/splits/training_subjects.txt",
            normalize=config.normalize
        )

        validation_dataset = RawDataset(
            h5_file_path=H5_FILE_PATH,
            subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/raw/splits/validation_subjects.txt",
            normalize=config.normalize
        )

        wandb.log({
            "training_dataset_size": len(training_dataset),
            "validation_dataset_size": len(validation_dataset),
        })

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)

        logger.success("Loader ready")
        
        model = Simple2D(
            n_filters=config.n_filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            dropout_rate=config.dropout_rate
        ).to(device)

        logger.success("Model ready")

        criterion = nn.BCEWithLogitsLoss()
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

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(config.max_epochs):
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
            with torch.no_grad():
                for data, target in validation_loader:
                    data, target = data.to(device), target.to(device).float()
                    outputs = model(data).squeeze(1)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == target).sum().item()
                    val_total += target.size(0)

            avg_val_loss = val_loss / len(validation_loader)
            avg_val_accuracy = val_correct / val_total

            train_time = time.time() - epoch_start_time
            if train_time > 60:
                logger.warning(f"Epoch {epoch+1} took {train_time:.2f} seconds, which is longer than a minute. Stopping training.")
                break

            logger.info(f"Epoch {epoch+1}/{config.max_epochs} completed in {train_time:.4f} seconds")
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

            if avg_val_loss < best_val_loss - config.min_delta:
                logger.success(f"New best validation loss: {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
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
                outputs = model(data).squeeze(1)
                loss = criterion(outputs, target)
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
        })

        logger.info(f"Final accuracy: {final_accuracy:.4f}")
        logger.info(f"Final F1 score: {final_f1:.4f}")
        logger.info(f"Final precision: {final_precision:.4f}")
        logger.info(f"Final recall: {final_recall:.4f}")
        logger.info(f"Final ROC AUC: {final_roc_auc:.4f}")
            
            
if __name__ == "__main__":
    main()