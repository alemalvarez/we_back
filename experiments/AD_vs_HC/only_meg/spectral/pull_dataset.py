import wandb

def pull_dataset_from_wandb() -> None:
    """
    Pulls the dataset artifact from Weights & Biases.
    Project: alejandro-mata-university-of-valladolid/dataset_creation
    Artifact: POCTEP_DK_features_only:v0
    """
    api = wandb.Api()
    artifact = api.artifact(
        "alejandro-mata-university-of-valladolid/dataset_creation/POCTEP_DK_features_only:v0",
        type="dataset"
    )
    artifact_dir = artifact.download()
    print(f"Dataset downloaded to: {artifact_dir}")

if __name__ == "__main__":
    pull_dataset_from_wandb()
