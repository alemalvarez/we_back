#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.schemas import OptimizerConfig, CriterionConfig, RunConfig
from models.spectral_net import SpectralNetConfig

print("Creating SpectralNetConfig...")
model_config = SpectralNetConfig()

print("\nCreating OptimizerConfig...")
optimizer_config = OptimizerConfig(
    learning_rate=3e-3,
    weight_decay=None,
)

print("\nCreating CriterionConfig...")
criterion_config = CriterionConfig(
    pos_weight_type='fixed',
    pos_weight_value=1.0,
)

print("\nCreating RunConfig...")
run_config = RunConfig(
    network_config=model_config,
    optimizer_config=optimizer_config,
    criterion_config=criterion_config,
    random_seed=42,
    batch_size=32,
    max_epochs=50,
    patience=1,
    min_delta=0.001,
    early_stopping_metric='mcc',
    normalization='standard',
    log_to_wandb=False,
    wandb_init=None,
)

print("\nDone.")
