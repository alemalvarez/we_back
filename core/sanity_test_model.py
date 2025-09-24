from collections.abc import Sized
import time
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
from core.schemas import BaseModelConfig
import torch.nn as nn
import torch.profiler
import torch.optim as optim

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

def sanity_test_model(
    model: nn.Module,
    config: BaseModelConfig,
    dataset: Dataset,
    run_overfit_test: bool = False,
    overfit_test_epochs: int = 100,
    )-> None:

    device = _get_device()
    model.to(device)
    logger.success("Model ready")
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    logger.success("Loader ready")

    criterion = nn.BCEWithLogitsLoss()
    one_batch = next(iter(loader))
    data, target = one_batch
    data, target = data.to(device), target.to(device).float()

    logger.info("--- Memory Estimations ---")
    # 1. Model Memory
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_params * 4 / (1024**2)  # Assuming float32 (4 bytes)
    logger.info(f"Model size (parameters): {model_size_mb:.2f} MB")

    # 2. Batch Memory
    batch_data_size_mb = data.nelement() * data.element_size() / (1024**2)
    batch_target_size_mb = target.nelement() * target.element_size() / (1024**2)
    total_batch_size_mb = batch_data_size_mb + batch_target_size_mb
    logger.info(f"One batch size (data + target): {total_batch_size_mb:.2f} MB")

    # 3. Full Dataset Memory
    assert isinstance(dataset, Sized), "Dataset must be a Sized object"
    num_samples = len(dataset)
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

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    profiler_log = prof.key_averages().table(sort_by=sort_key, row_limit=15)
    logger.info(f"\nProfiler results (sorted by {sort_key}):\n" + profiler_log)
    logger.info("The table above shows memory usage in the 'Mem' columns.")

    key_averages = prof.key_averages()
    model_inference_event = next((e for e in key_averages if e.key == "model_inference"), None)

    if model_inference_event:
        time_per_batch_us = 0.0
        if device.type == 'cuda':
            time_per_batch_us = model_inference_event.cuda_time_total
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
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
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