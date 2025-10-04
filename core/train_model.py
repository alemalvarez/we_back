from collections.abc import Sized
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from core.schemas import BaseModelConfig
from loguru import logger
from collections import defaultdict
from core.training_loop import one_epoch, OneEpochResults
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    matthews_corrcoef,
)
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
        shuffle=True,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    logger.success("Loader ready")
    
    if hasattr(config, "pos_weight") and config.pos_weight is not None:
        weight = torch.tensor(config.pos_weight)
    else:
        weight = torch.tensor(train_pos / train_neg)

    logger.info(f"Weight: {weight}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    if hasattr(config, "weight_decay") and config.weight_decay is not None:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create scheduler if cosine annealing is enabled
    scheduler = None
    if getattr(config, "use_cosine_annealing", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-6
        )
        logger.success("Optimizer and cosine annealing scheduler ready")
    else:
        logger.success("Optimizer ready")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f"Total parameters: {total_params}")
    logger.debug(f"Trainable parameters: {trainable_params}")

    best_early_stopping_metric = float('inf') if config.early_stopping_metric == 'loss' else -float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.max_epochs):
        results: OneEpochResults = one_epoch(
            model, 
            criterion, 
            optimizer, 
            train_loader, 
            validation_loader, 
            device,
        )

        logger.info(f"Epoch {epoch+1}/{config.max_epochs} completed in {results.epoch_time:.4f} seconds")
        logger.info(f"Train Loss: {results.train_loss:.4f}, Train Accuracy: {results.train_accuracy:.2f}")
        logger.info(f"Validation Loss: {results.val_loss:.4f}, Validation Accuracy: {results.val_accuracy:.2f}")
        logger.info(f"Validation f1: {results.val_f1:.4f}, Validation mcc: {results.val_mcc:.4f}, Validation kappa: {results.val_kappa:.4f}")
        
        # Early stopping check
        if config.early_stopping_metric == 'loss':
            early_stopping_metric = results.val_loss
            is_better = early_stopping_metric < best_early_stopping_metric - config.min_delta
        elif config.early_stopping_metric == 'f1':
            early_stopping_metric = results.val_f1
            is_better = early_stopping_metric > best_early_stopping_metric + config.min_delta
        elif config.early_stopping_metric == 'mcc':
            early_stopping_metric = results.val_mcc
            is_better = early_stopping_metric > best_early_stopping_metric + config.min_delta
        elif config.early_stopping_metric == 'kappa':
            early_stopping_metric = results.val_kappa
            is_better = early_stopping_metric > best_early_stopping_metric + config.min_delta
            
        if is_better:
            best_early_stopping_metric = early_stopping_metric
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.success(f"New best validation {config.early_stopping_metric}: {early_stopping_metric:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Step the scheduler at the end of each epoch if enabled
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate after annealing: {current_lr:.6f}")

        logger.info(100*"=")
        
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
    y_pred_proba_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device).float()
            logits = model(data).squeeze(1)
            probabilities = torch.sigmoid(logits)
            y_pred_proba_list.extend(probabilities.cpu().numpy())
            y_true_list.extend(target.cpu().numpy())

    y_pred_proba = np.array(y_pred_proba_list)
    y_true = np.array(y_true_list)
    logger.info(f"Collected {len(y_true)} predictions from validation set")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate MCC for each threshold
    mccs = []
    for threshold in thresholds:
        y_pred_at_threshold = (y_pred_proba > threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_at_threshold)
        mccs.append(mcc)

    best_idx = np.argmax(mccs)
    best_threshold = thresholds[best_idx]
    logger.info(f"Best threshold: {best_threshold:.4f}")

    y_pred = (y_pred_proba > best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    logger.info(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    assert hasattr(validation_dataset, "sample_to_subject"), "Validation dataset must have a sample_to_subject attribute"
    sample_to_subject = validation_dataset.sample_to_subject  # type: ignore
    assert len(sample_to_subject) == len(y_true), "sample_to_subject length must match number of validation samples"
    subject_correct: defaultdict[str, int] = defaultdict(int)
    subject_wrong: defaultdict[str, int] = defaultdict(int)
    for idx, subject in enumerate(sample_to_subject):
        if y_pred[idx] == y_true[idx]:
            subject_correct[subject] += 1
        else:
            subject_wrong[subject] += 1
    for subject in sorted(set(sample_to_subject)):
        logger.info(f"Subject {subject}: correct={subject_correct[subject]}, wrong={subject_wrong[subject]}")

    final_accuracy = accuracy_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_roc_auc = roc_auc_score(y_true, y_pred_proba)
    final_mcc = matthews_corrcoef(y_true, y_pred)
    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    logger.info(f"Final F1 score: {final_f1:.4f}")
    logger.info(f"Final precision: {final_precision:.4f}")
    logger.info(f"Final recall: {final_recall:.4f}")
    logger.info(f"Final ROC AUC: {final_roc_auc:.4f}")
    logger.info(f"Final MCC: {final_mcc:.4f}")
