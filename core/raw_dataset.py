from typing import List, Literal, Tuple, Optional
import h5py  # type: ignore
from loguru import logger
from torch.utils.data import Dataset
import numpy as np
import torch


class RawDataset(Dataset):
    """PyTorch dataset for raw EEG segments."""

    samples: torch.Tensor
    labels: torch.Tensor
    sample_to_subject: List[str]
    augment: bool
    augment_prob: Tuple[float, float]
    noise_std: float

    def __init__(
        self, 
        h5_file_path: str, 
        subjects_txt_path: Optional[str] = None,
        normalize: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full'] = 'sample-channel',
        augment: bool = False,
        augment_prob: Tuple[float, float] = (0.5, 0.0), # neg, pos
        noise_std: float = 0.1,
        subjects_list: Optional[List[str]] = None,
    ):
        if subjects_list is not None:
            logger.info(f"Using subjects_list with {len(subjects_list)} subjects")
            self.subject_ids = subjects_list
        else:
            assert subjects_txt_path is not None, "subjects_txt_path must be provided if subjects_list is not provided"
            logger.info(f"Loading raw segments dataset from {h5_file_path} with {len(self.subject_ids)} subjects")
            with open(subjects_txt_path, 'r') as f:
                self.subject_ids = [line.strip() for line in f.readlines()]

        # Collect subject data and prepare for normalization
        features_list = []
        labels_list = []
        self.sample_to_subject: List[str] = []
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        subject_data = {}  # Store subject data for normalization
        all_segments = None  # Will be created only when needed

        with h5py.File(h5_file_path, 'r') as f:
            for subj_key in self.subject_ids:
                subject = f['subjects'][subj_key]
                n_segments = subject.attrs['n_segments']
                raw_segments = subject['raw_segments'][()]

                subject_data[subj_key] = {
                    'data': raw_segments,
                    'category': subject.attrs['category'],
                    'n_segments': n_segments
                }

        # Create all_segments only when needed for global normalization
        if normalize in ['channel', 'full']:
            all_data = [subject_data[subj_key]['data'] for subj_key in self.subject_ids]
            all_segments = np.concatenate(all_data, axis=0)
        
        # Apply normalization based on scope
        if normalize == 'sample-channel':
            # Normalize each segment and each channel independently
            normalized_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['data']
                # data shape: (n_segments, n_samples, n_channels)
                mean_vals = data.mean(axis=1, keepdims=True, dtype=np.float32)  # mean per segment per channel
                std_vals = data.std(axis=1, keepdims=True, dtype=np.float32)    # std per segment per channel
                std_vals[std_vals == 0] = 1.0
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_data.append(normalized_segments)
                
        elif normalize == 'sample':
            # Normalize each segment across all channels
            normalized_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['data']
                # Flatten last two dimensions for mean/std calculation
                data_flat = data.reshape(data.shape[0], -1)  # (n_segments, n_samples * n_channels)
                mean_vals = data_flat.mean(axis=1, keepdims=True, dtype=np.float32)  # mean per segment
                std_vals = data_flat.std(axis=1, keepdims=True, dtype=np.float32)    # std per segment
                std_vals[std_vals == 0] = 1.0
                # Reshape back and normalize
                mean_vals = mean_vals.reshape(data.shape[0], 1, 1)
                std_vals = std_vals.reshape(data.shape[0], 1, 1)
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_data.append(normalized_segments)
                
        elif normalize == 'channel-subject':
            # Normalize per subject per channel across all segments
            normalized_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['data']
                # data shape: (n_segments, n_samples, n_channels)
                mean_vals = data.mean(axis=(0, 1), keepdims=True, dtype=np.float32)  # mean per channel across all segments and samples
                std_vals = data.std(axis=(0, 1), keepdims=True, dtype=np.float32)    # std per channel across all segments and samples
                std_vals[std_vals == 0] = 1.0
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_data.append(normalized_segments)
                
        elif normalize == 'subject':
            # Normalize per subject across all segments and channels
            normalized_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['data']
                mean_val = data.mean(dtype=np.float32)  # single mean for entire subject
                std_val = data.std(dtype=np.float32)    # single std for entire subject
                if std_val == 0:
                    std_val = 1.0
                normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                normalized_data.append(normalized_segments)
                
        elif normalize == 'channel':
            # Normalize per channel across entire dataset
            if all_segments is None:
                raise ValueError("all_segments should not be None for channel normalization")
            mean_vals = all_segments.mean(axis=(0, 1), keepdims=True, dtype=np.float32)  # mean per channel across all data
            std_vals = all_segments.std(axis=(0, 1), keepdims=True, dtype=np.float32)    # std per channel across all data
            std_vals[std_vals == 0] = 1.0

            normalized_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['data']
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_data.append(normalized_segments)

        elif normalize == 'full':
            # Normalize across entire dataset
            if all_segments is None:
                raise ValueError("all_segments should not be None for full normalization")
            mean_val = all_segments.mean(dtype=np.float32)  # single mean for entire dataset
            std_val = all_segments.std(dtype=np.float32)    # single std for entire dataset
            if std_val == 0:
                std_val = 1.0

            normalized_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['data']
                normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                normalized_data.append(normalized_segments)

        else:
            logger.warning(f"No normalization, or normalization invalid: {normalize}")
            logger.warning("Using no normalization")
            normalized_data = [subject_data[subj_key]['data'] for subj_key in self.subject_ids]
        
        # Build final features and labels lists
        for subj_idx, subj_key in enumerate(self.subject_ids):
            data = normalized_data[subj_idx]
            n_segments = subject_data[subj_key]['n_segments']
            category = subject_data[subj_key]['category']
            
            for seg_idx in range(n_segments):
                features_list.append(data[seg_idx, :, :])
                labels_list.append(0 if category == 'HC' else 1)
                self.sample_to_subject.append(subj_key)

        # Convert to torch tensors with proper data types
        self.samples = torch.tensor(np.array(features_list), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels_list), dtype=torch.long)

        logger.debug(f"Mean after normalization: {self.samples.mean()}")
        logger.debug(f"Std after normalization: {self.samples.std()}")


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = self.samples[idx].clone()
        y = self.labels[idx]
        if (
            self.augment and 
            torch.rand(1).item() < self.augment_prob[y.item()]
        ):
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        return x, y

    def get_sample_to_subject(self, idx: int) -> str:
        return self.sample_to_subject[idx]

if __name__ == "__main__":
    h5_file = "h5test_raw_only.h5"
    subjects_file = "experiments/ADSEV_vs_HC/POCTEP/raw/splits/training_subjects.txt"
    
    NormScope = Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    normalization_scopes: List[NormScope] = ['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']
    
    print("Testing all normalization scopes:")
    print("=" * 50)
    
    for norm_scope in normalization_scopes:
        print(f"\n--- Testing {norm_scope} normalization ---")
        
        try:
            dataset = RawDataset(
                h5_file_path=h5_file, 
                subjects_txt_path=subjects_file,
                normalize=norm_scope
            )
            
            print(f"Dataset length: {len(dataset)}")
            print(f"Samples shape: {dataset.samples.shape}")
            print(f"Labels shape: {dataset.labels.shape}")

            # Overall stats
            overall_mean = dataset.samples.mean().item()
            overall_std = dataset.samples.std().item()
            print(f"Overall mean: {overall_mean:.6f}")
            print(f"Overall std: {overall_std:.6f}")

            # First subject overall stats
            first_subject = dataset.sample_to_subject[0]
            first_subject_indices = [i for i, s in enumerate(dataset.sample_to_subject) if s == first_subject]
            print(f"first_subject_indices shape: {len(first_subject_indices)}")
            first_subject_samples = dataset.samples[first_subject_indices]
            print(f"first_subject_samples shape: {first_subject_samples.shape}")
            first_subject_mean = first_subject_samples.mean().item()
            first_subject_std = first_subject_samples.std().item()
            print(f"First subject overall mean: {first_subject_mean:.6f}")
            print(f"First subject overall std: {first_subject_std:.6f}")

            # First segment on first subject (overall)
            first_segment = first_subject_samples[0]
            print(f"first_segment shape: {first_segment.shape}")
            first_segment_mean = first_segment.mean().item()
            first_segment_std = first_segment.std().item()
            print(f"First segment on first subject mean: {first_segment_mean:.6f}")
            print(f"First segment on first subject std: {first_segment_std:.6f}")

            # First channel on first segment on first subject
            first_channel = first_segment[:, 0]
            print(f"first_channel shape: {first_channel.shape}")
            first_channel_mean = first_channel.mean().item()
            first_channel_std = first_channel.std().item()
            print(f"First channel on first segment on first subject mean: {first_channel_mean:.6f}")
            print(f"First channel on first segment on first subject std: {first_channel_std:.6f}")

            # Also test another channel to verify per-channel normalization
            if first_segment.shape[1] > 1:
                second_channel = first_segment[:, 1]
                print(f"second_channel shape: {second_channel.shape}")
                second_channel_mean = second_channel.mean().item()
                second_channel_std = second_channel.std().item()
                print(f"Second channel on first segment on first subject mean: {second_channel_mean:.6f}")
                print(f"Second channel on first segment on first subject std: {second_channel_std:.6f}")

            # Let's also check if the normalization is working by testing multiple segments
            if len(first_subject_indices) > 1:
                second_segment = first_subject_samples[1]
                print(f"second_segment shape: {second_segment.shape}")
                second_segment_first_channel = second_segment[:, 0]
                print(f"second_segment_first_channel shape: {second_segment_first_channel.shape}")
                second_segment_first_channel_mean = second_segment_first_channel.mean().item()
                second_segment_first_channel_std = second_segment_first_channel.std().item()
                print(f"First channel on second segment on first subject mean: {second_segment_first_channel_mean:.6f}")
                print(f"First channel on second segment on first subject std: {second_segment_first_channel_std:.6f}")

        except Exception as e:
            print(f"Error with {norm_scope} normalization: {str(e)}")
    
    print("\n" + "=" * 50)
    print("All normalization tests completed!")