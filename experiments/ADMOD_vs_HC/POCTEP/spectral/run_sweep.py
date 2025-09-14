#!/usr/bin/env python3
"""
Script to run a wandb sweep for the SpectralNet neural network.
"""

import wandb
import os

def main():
    # Navigate to the correct directory
    os.chdir("/Users/alemalvarez/code-workspace/TFG/we_back")

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config="/Users/alemalvarez/code-workspace/TFG/we_back/experiments/ADMOD_vs_HC/POCTEP/spectral/sweep_config.yaml",
        project="ADMOD_vs_HC_sweep"
    )

    print(f"Sweep created with ID: {sweep_id}")
    print("To start the sweep, run:")
    print(f"wandb agent {sweep_id}")
    print()
    print("Or run multiple agents in parallel:")
    print(f"wandb agent {sweep_id} & wandb agent {sweep_id} & wandb agent {sweep_id}")

if __name__ == "__main__":
    main()

