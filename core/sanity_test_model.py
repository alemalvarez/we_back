from collections.abc import Sized
import time
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
from core.schemas import RunConfig
import torch.profiler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score # type: ignore
from core.builders import build_model, build_optimizer, build_criterion

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
    pos = 0
    neg = 0
    for i in range(len(dataset)): # type: ignore
        sample = dataset[i]
        label = sample[-1]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if int(label) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg

def _unpack_batch(batch: tuple, device: torch.device) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Unpack batch and move to device. Handles both (x, y) and ((x_raw, x_spectral), y) formats."""
    data_or_tuple, target = batch  # type: ignore
    
    if isinstance(data_or_tuple, (tuple, list)):
        data: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = tuple(d.to(device) for d in data_or_tuple)
    else:
        data = data_or_tuple.to(device)
    
    target = target.to(device).float()
    return data, target

def sanity_test_model(
    config: RunConfig,
    training_dataset: Dataset,
    run_overfit_test: bool = False,
    overfit_test_epochs: int = 100,
    )-> None:

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    device = _get_device()
    
    model = build_model(config.network_config)
    model = model.to(device)
    logger.success("Model ready")
    
    optimizer = build_optimizer(config.optimizer_config, model)
    criterion = build_criterion(config.criterion_config, _count_pos_neg(training_dataset))
    
    loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
    logger.success("Loader ready")

    one_batch = next(iter(loader))
    data, target = _unpack_batch(one_batch, device)

    logger.info("--- Memory Estimations ---")
    # 1. Model Memory
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_params * 4 / (1024**2)  # Assuming float32 (4 bytes)
    logger.info(f"Model size (parameters): {model_size_mb:.2f} MB")

    # 2. Batch Memory
    if isinstance(data, tuple):
        batch_data_size_mb = sum(d.nelement() * d.element_size() for d in data) / (1024**2)
    else:
        batch_data_size_mb = data.nelement() * data.element_size() / (1024**2)
    batch_target_size_mb = target.nelement() * target.element_size() / (1024**2)
    total_batch_size_mb = batch_data_size_mb + batch_target_size_mb
    logger.info(f"One batch size (data + target): {total_batch_size_mb:.2f} MB")

    # 3. Full Dataset Memory
    assert isinstance(training_dataset, Sized), "Dataset must be a Sized object"
    num_samples = len(training_dataset)
    size_of_one_sample_mb = total_batch_size_mb / config.batch_size
    estimated_dataset_size_mb = size_of_one_sample_mb * num_samples
    logger.info(f"Estimated dataset size ({num_samples} samples): {estimated_dataset_size_mb:.2f} MB")
    logger.info("--------------------------")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            outputs = model(data).squeeze(1)

        loss = criterion(outputs, target)
        logger.success(f"Initial loss on one batch: {loss.item():.4f}")

    model.eval()
    y_pred_list: list[np.ndarray] = []
    y_pred_proba_list: list[np.ndarray] = []
    y_true_list: list[np.ndarray] = []
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in loader:
            data, target = _unpack_batch(batch, device)
            outputs = model(data).squeeze(1)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == target).sum().item()
            val_total += target.size(0)

            y_pred_proba_list.extend(outputs.cpu().numpy())
            predictions = (outputs > 0.5).float()
            y_pred_list.extend(predictions.cpu().numpy())
            y_true_list.extend(target.cpu().numpy())

    avg_val_loss = val_loss / len(loader)
    avg_val_accuracy = val_correct / val_total
    
    logger.success(f"Final loss on validation set: {avg_val_loss:.4f}")
    logger.success(f"Final accuracy on validation set: {avg_val_accuracy:.4f}")

    y_pred = np.array(y_pred_list)
    y_pred_proba = np.array(y_pred_proba_list)
    y_true = np.array(y_true_list)

    logger.debug(y_pred)
    logger.debug(y_true)

    final_accuracy = accuracy_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_roc_auc = roc_auc_score(y_true, y_pred_proba)

    logger.success(f"Final accuracy: {final_accuracy:.4f}")
    logger.success(f"Final F1 score: {final_f1:.4f}")
    logger.success(f"Final precision: {final_precision:.4f}")
    logger.success(f"Final recall: {final_recall:.4f}")
    logger.success(f"Final ROC AUC: {final_roc_auc:.4f}")

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    profiler_log = prof.key_averages().table(sort_by=sort_key, row_limit=15)
    logger.info(f"\nProfiler results (sorted by {sort_key}):\n" + profiler_log)
    logger.info("The table above shows memory usage in the 'Mem' columns.")

    key_averages = prof.key_averages()
    model_inference_event = next((e for e in key_averages if e.key == "model_inference"), None)

    if model_inference_event:
        time_per_batch_us = 0.0
        if device.type == 'cuda':
            time_per_batch_us = model_inference_event.device_time_total
        else:
            time_per_batch_us = model_inference_event.cpu_time_total
        
        time_per_batch_s = time_per_batch_us / 1_000_000
        total_batches = len(loader)
        estimated_epoch_time_s = time_per_batch_s * total_batches

        logger.success(
            f"Estimated time per epoch: {estimated_epoch_time_s:.2f} seconds ({estimated_epoch_time_s/60:.2f} minutes)"
        )
        logger.info(
            f"Based on {time_per_batch_s * 1000:.2f} ms/batch for a batch of size {config.batch_size}."
        )
    else:
        logger.warning(
            "Could not find 'model_inference' event in profiler results. Cannot estimate epoch time."
        )

    if run_overfit_test:
        model.train()
        logger.info("Running an overfitting test on a single batch...")
        data_overfit, target_overfit = data, target
        for epoch in range(overfit_test_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()
            outputs_overfit = model(data_overfit).squeeze(1)
            loss_overfit = criterion(outputs_overfit, target_overfit)
            loss_overfit.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            optimizer.step()
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % 10 == 0:
                logger.info(f"Overfit Epoch [{epoch+1}/{overfit_test_epochs}], Loss: {loss_overfit.item():.6f}, Grad Norm: {total_norm:.4f}, Time: {epoch_time:.4f} seconds")

        logger.success(
            f"Overfitting test completed. Final loss: {loss_overfit.item():.6f}. "
            "If the loss has significantly decreased, the model is likely learning correctly."
        )