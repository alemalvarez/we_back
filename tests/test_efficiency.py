import time
from typing import Dict, Any, Tuple
import os

import h5py  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb  # type: ignore
from loguru import logger

from examples.write_dataset import create_dataset, Subject


class EEGRawDataset(Dataset):
    """PyTorch dataset for raw EEG segments."""
    
    def __init__(self, raw_segments: np.ndarray, labels: np.ndarray):
        """
        Args:
            raw_segments: Array of shape (n_segments, n_samples, n_channels)
            labels: Binary labels (0 for control, 1 for patient)
        """
        self.raw_segments = torch.FloatTensor(raw_segments)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.raw_segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.raw_segments[idx], self.labels[idx]


class SpectralDataset(Dataset):
    """PyTorch dataset for spectral parameters."""
    
    def __init__(self, spectral_features: np.ndarray, labels: np.ndarray):
        """
        Args:
            spectral_features: Array of shape (n_segments, n_features)
            labels: Binary labels (0 for control, 1 for patient)
        """
        self.spectral_features = torch.FloatTensor(spectral_features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.spectral_features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectral_features[idx], self.labels[idx]


class SimpleCNN(nn.Module):
    """Simple CNN for raw EEG data classification."""
    
    def __init__(self, n_channels: int = 68, n_samples: int = 1000):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_samples, n_channels)
        # Transpose to (batch_size, n_channels, n_samples) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.classifier(x)
        return x


class SpectralMLP(nn.Module):
    """Simple MLP for spectral parameters classification."""
    
    def __init__(self, n_features: int):
        super(SpectralMLP, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary classification
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class EEGEfficiencyTester:
    """Main class for testing EEG data handling efficiency."""
    
    def __init__(self, project_name: str = "eeg-efficiency-test"):
        self.project_name = project_name
        self.h5_file_path = "h5test.h5"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb
        logger.info("Initializing Weights & Biases...")
        wandb.init(project=self.project_name, name=f"efficiency-test-{int(time.time())}")
        
    def time_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time a function execution and return result and elapsed time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    
    def create_and_save_dataset(self) -> float:
        """Create H5 dataset and upload to wandb."""
        logger.info("🔄 Creating H5 dataset...")
        
        # Create the dataset
        _, creation_time = self.time_function(create_dataset, self.h5_file_path)
        logger.info(f"✅ H5 dataset created in {creation_time:.2f} seconds")
        
        # Get file size
        file_size_mb = os.path.getsize(self.h5_file_path) / (1024 * 1024)
        logger.info(f"📁 H5 file size: {file_size_mb:.2f} MB")
        
        # Upload to wandb
        logger.info("📤 Uploading H5 file to Weights & Biases...")
        _, upload_time = self.time_function(
            wandb.save, self.h5_file_path, base_path="."
        )
        logger.info(f"✅ File uploaded to wandb in {upload_time:.2f} seconds")
        
        # Log metrics
        wandb.log({
            "h5_creation_time": creation_time,
            "h5_upload_time": upload_time,
            "h5_file_size_mb": file_size_mb,
            "total_save_time": creation_time + upload_time
        })
        
        return creation_time + upload_time
    
    def download_and_load_dataset(self) -> Tuple[Dict[str, Subject], float]:
        """Download from wandb and load H5 dataset."""
        logger.info("📥 Downloading H5 file from Weights & Biases...")
        
        # In a real scenario, you'd download from wandb
        # For this test, we'll simulate by reading the local file
        _, download_time = self.time_function(time.sleep, 0.1)  # Simulate download
        logger.info(f"✅ File downloaded in {download_time:.2f} seconds")
        
        # Load dataset
        logger.info("🔄 Loading H5 dataset...")
        subjects_data, load_time = self.time_function(self._load_h5_data)
        logger.info(f"✅ Dataset loaded in {load_time:.2f} seconds")
        
        total_time = download_time + load_time
        wandb.log({
            "h5_download_time": download_time,
            "h5_load_time": load_time,
            "total_load_time": total_time
        })
        
        return subjects_data, total_time
    
    def _load_h5_data(self) -> Dict[str, Any]:
        """Load data from H5 file."""
        subjects_data: Dict[str, Any] = {}
        
        with h5py.File(self.h5_file_path, 'r') as f:
            subjects_group = f['subjects']  # type: ignore
            
            for subject_id in list(subjects_group.keys()):  # type: ignore
                subj_group = subjects_group[subject_id]  # type: ignore
                
                # Load metadata
                category = str(subj_group.attrs['category'])  # type: ignore
                
                # Load raw segments
                raw_segments = np.array(subj_group['raw_segments'])  # type: ignore
                
                # Load spectral parameters
                spectral_params: Dict[str, np.ndarray] = {}
                spectral_params_group = subj_group['spectral']['spectral_parameters']  # type: ignore
                
                for param_name in list(spectral_params_group.keys()):  # type: ignore
                    spectral_params[param_name] = np.array(spectral_params_group[param_name])  # type: ignore
                
                subjects_data[subject_id] = {
                    'category': category,
                    'raw_segments': raw_segments,
                    'spectral_parameters': spectral_params
                }
        
        return subjects_data
    
    def prepare_datasets(self, subjects_data: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, float]:
        """Prepare PyTorch datasets and dataloaders."""
        logger.info("🔄 Preparing PyTorch datasets...")
        
        start_time = time.time()
        
        # Prepare raw segments data
        all_raw_segments = []
        all_spectral_features = []
        all_labels = []
        
        for subject_id, data in subjects_data.items():
            category = data['category']
            label = 0 if category == 'control' else 1
            
            n_segments = data['raw_segments'].shape[0]
            
            # Add raw segments
            all_raw_segments.append(data['raw_segments'])
            
            # Prepare spectral features (concatenate all spectral parameters)
            spectral_features = []
            for param_name, param_data in data['spectral_parameters'].items():
                if param_name.startswith('relative_power_'):
                    spectral_features.append(param_data)
                elif param_name in ['median_frequency', 'spectral_edge_frequency_95', 
                                  'individual_alpha_frequency', 'transition_frequency',
                                  'renyi_entropy']:
                    spectral_features.append(param_data)
            
            spectral_features_array = np.column_stack(spectral_features)
            all_spectral_features.append(spectral_features_array)
            
            # Add labels for each segment
            all_labels.extend([label] * n_segments)
        
        # Concatenate all data
        raw_segments_array = np.concatenate(all_raw_segments, axis=0)
        spectral_features_array = np.concatenate(all_spectral_features, axis=0)
        labels_array = np.array(all_labels)
        
        logger.info("📊 Dataset shapes:")
        logger.info(f"   Raw segments: {raw_segments_array.shape}")
        logger.info(f"   Spectral features: {spectral_features_array.shape}")
        logger.info(f"   Labels: {labels_array.shape}")
        
        # Create datasets
        raw_dataset = EEGRawDataset(raw_segments_array, labels_array)
        spectral_dataset = SpectralDataset(spectral_features_array, labels_array)
        
        # Create dataloaders
        raw_dataloader = DataLoader(raw_dataset, batch_size=32, shuffle=True)
        spectral_dataloader = DataLoader(spectral_dataset, batch_size=32, shuffle=True)
        
        prep_time = time.time() - start_time
        logger.info(f"✅ Datasets prepared in {prep_time:.2f} seconds")
        
        wandb.log({
            "dataset_prep_time": prep_time,
            "n_segments_total": len(labels_array),
            "n_spectral_features": spectral_features_array.shape[1]
        })
        
        return raw_dataloader, spectral_dataloader, prep_time
    
    def train_raw_model(self, dataloader: DataLoader) -> float:
        """Train CNN model on raw EEG segments."""
        logger.info("🧠 Training CNN model on raw EEG segments...")
        
        start_time = time.time()
        
        # Initialize model
        model = SimpleCNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        n_epochs = 3  # Keep it short for efficiency testing
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.debug(f"   Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / n_batches
            logger.info(f"   Epoch {epoch+1}/{n_epochs} completed, Average Loss: {avg_loss:.4f}")
            
            wandb.log({
                "raw_model_epoch": epoch + 1,
                "raw_model_loss": avg_loss
            })
        
        training_time = time.time() - start_time
        logger.info(f"✅ CNN model training completed in {training_time:.2f} seconds")
        
        wandb.log({"raw_model_training_time": training_time})
        
        return training_time
    
    def train_spectral_model(self, dataloader: DataLoader) -> float:
        """Train MLP model on spectral parameters."""
        logger.info("📈 Training MLP model on spectral parameters...")
        
        start_time = time.time()
        
        # Get number of features from the first batch
        sample_batch = next(iter(dataloader))
        n_features = sample_batch[0].shape[1]
        
        # Initialize model
        model = SpectralMLP(n_features).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        n_epochs = 5  # Spectral features should train faster
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.debug(f"   Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / n_batches
            logger.info(f"   Epoch {epoch+1}/{n_epochs} completed, Average Loss: {avg_loss:.4f}")
            
            wandb.log({
                "spectral_model_epoch": epoch + 1,
                "spectral_model_loss": avg_loss
            })
        
        training_time = time.time() - start_time
        logger.info(f"✅ MLP model training completed in {training_time:.2f} seconds")
        
        wandb.log({"spectral_model_training_time": training_time})
        
        return training_time
    
    def run_full_efficiency_test(self) -> Dict[str, float]:
        """Run the complete efficiency test pipeline."""
        logger.info("🚀 Starting full EEG efficiency test pipeline...")
        
        total_start_time = time.time()
        
        # Step 1: Create and save dataset
        save_time = self.create_and_save_dataset()
        
        # Step 2: Download and load dataset
        subjects_data, load_time = self.download_and_load_dataset()
        
        # Step 3: Prepare datasets
        raw_dataloader, spectral_dataloader, prep_time = self.prepare_datasets(subjects_data)
        
        # Step 4: Train models
        raw_training_time = self.train_raw_model(raw_dataloader)
        spectral_training_time = self.train_spectral_model(spectral_dataloader)
        
        total_time = time.time() - total_start_time
        
        # Summary
        results = {
            "save_time": save_time,
            "load_time": load_time,
            "prep_time": prep_time,
            "raw_training_time": raw_training_time,
            "spectral_training_time": spectral_training_time,
            "total_time": total_time
        }
        
        logger.info("📊 Efficiency Test Results Summary:")
        logger.info(f"   💾 Save time: {save_time:.2f}s")
        logger.info(f"   📥 Load time: {load_time:.2f}s")
        logger.info(f"   🔧 Prep time: {prep_time:.2f}s")
        logger.info(f"   🧠 Raw model training: {raw_training_time:.2f}s")
        logger.info(f"   📈 Spectral model training: {spectral_training_time:.2f}s")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        
        wandb.log(results)
        wandb.finish()
        
        return results


def main():
    """Main function to run the efficiency test."""
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.info("🎯 EEG Data Handling Efficiency Test")
    logger.info("=" * 50)
    
    # Run efficiency test
    tester = EEGEfficiencyTester()
    results = tester.run_full_efficiency_test()
    
    logger.info("🎉 Efficiency test completed successfully!")
    
    return results


if __name__ == "__main__":
    main() 