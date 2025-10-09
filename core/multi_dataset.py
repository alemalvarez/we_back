from typing import List, Literal, Tuple, Optional
import h5py  # type: ignore
from loguru import logger
from torch.utils.data import Dataset
import numpy as np
import torch


class MultiDataset(Dataset):
    """PyTorch dataset combining raw EEG segments and spectral features."""

    raw_samples: torch.Tensor
    spectral_features: torch.Tensor
    labels: torch.Tensor
    sample_to_subject: List[str]

    def __init__(
        self, 
        h5_file_path: str, 
        subjects_txt_path: Optional[str] = None,
        normalize_raw: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full'] = 'sample-channel',
        normalize_spectral: Literal['min-max', 'standard', 'none'] = 'standard',
        subjects_list: Optional[List[str]] = None,
    ):
        if subjects_list is not None:
            logger.info(f"Using subjects_list with {len(subjects_list)} subjects")
            self.subject_ids = subjects_list
        else:
            assert subjects_txt_path is not None, "subjects_txt_path must be provided if subjects_list is not provided"
            with open(subjects_txt_path, 'r') as f:
                self.subject_ids = [line.strip() for line in f.readlines()]
            logger.info(f"Loading combined dataset from {h5_file_path} with {len(self.subject_ids)} subjects")

        # Collect data
        raw_features_list = []
        spectral_features_list = []
        labels_list = []
        self.sample_to_subject: List[str] = []
        subject_data = {}

        with h5py.File(h5_file_path, 'r') as f:
            for subj_key in self.subject_ids:
                subject = f['subjects'][subj_key]
                n_segments = subject.attrs['n_segments']
                raw_segments = subject['raw_segments'][()]
                
                subject_data[subj_key] = {
                    'raw_data': raw_segments,
                    'category': subject.attrs['category'],
                    'n_segments': n_segments
                }

        # Normalize raw data
        all_segments = None
        if normalize_raw in ['channel', 'full']:
            all_data = [subject_data[subj_key]['raw_data'] for subj_key in self.subject_ids]
            all_segments = np.concatenate(all_data, axis=0)
        
        if normalize_raw == 'sample-channel':
            normalized_raw_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['raw_data']
                mean_vals = data.mean(axis=1, keepdims=True, dtype=np.float32)
                std_vals = data.std(axis=1, keepdims=True, dtype=np.float32)
                std_vals[std_vals == 0] = 1.0
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_raw_data.append(normalized_segments)
                
        elif normalize_raw == 'sample':
            normalized_raw_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['raw_data']
                data_flat = data.reshape(data.shape[0], -1)
                mean_vals = data_flat.mean(axis=1, keepdims=True, dtype=np.float32)
                std_vals = data_flat.std(axis=1, keepdims=True, dtype=np.float32)
                std_vals[std_vals == 0] = 1.0
                mean_vals = mean_vals.reshape(data.shape[0], 1, 1)
                std_vals = std_vals.reshape(data.shape[0], 1, 1)
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_raw_data.append(normalized_segments)
                
        elif normalize_raw == 'channel-subject':
            normalized_raw_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['raw_data']
                mean_vals = data.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
                std_vals = data.std(axis=(0, 1), keepdims=True, dtype=np.float32)
                std_vals[std_vals == 0] = 1.0
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_raw_data.append(normalized_segments)
                
        elif normalize_raw == 'subject':
            normalized_raw_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['raw_data']
                mean_val = data.mean(dtype=np.float32)
                std_val = data.std(dtype=np.float32)
                if std_val == 0:
                    std_val = 1.0
                normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                normalized_raw_data.append(normalized_segments)
                
        elif normalize_raw == 'channel':
            if all_segments is None:
                raise ValueError("all_segments should not be None for channel normalization")
            mean_vals = all_segments.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
            std_vals = all_segments.std(axis=(0, 1), keepdims=True, dtype=np.float32)
            std_vals[std_vals == 0] = 1.0
            normalized_raw_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['raw_data']
                normalized_segments = (data.astype(np.float32) - mean_vals) / std_vals
                normalized_raw_data.append(normalized_segments)

        elif normalize_raw == 'full':
            if all_segments is None:
                raise ValueError("all_segments should not be None for full normalization")
            mean_val = all_segments.mean(dtype=np.float32)
            std_val = all_segments.std(dtype=np.float32)
            if std_val == 0:
                std_val = 1.0
            normalized_raw_data = []
            for subj_key in self.subject_ids:
                data = subject_data[subj_key]['raw_data']
                normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                normalized_raw_data.append(normalized_segments)
        else:
            normalized_raw_data = [subject_data[subj_key]['raw_data'] for subj_key in self.subject_ids]
        
        # Extract spectral features and build lists
        with h5py.File(h5_file_path, 'r') as f:
            for subj_idx, subj_key in enumerate(self.subject_ids):
                subject = f['subjects'][subj_key]
                n_segments = subject_data[subj_key]['n_segments']
                category = subject_data[subj_key]['category']
                raw_data = normalized_raw_data[subj_idx]
                
                spectral_params = subject['spectral/spectral_parameters']
                
                for seg_idx in range(n_segments):
                    # Add raw data
                    raw_features_list.append(raw_data[seg_idx, :, :])
                    
                    # Add spectral features
                    relative_powers = spectral_params['relative_powers'][seg_idx, :]
                    features = [
                        float(spectral_params['individual_alpha_frequency'][seg_idx]),
                        float(spectral_params['median_frequency'][seg_idx]),
                        float(relative_powers[0]),  # Delta
                        float(relative_powers[1]),  # Theta
                        float(relative_powers[2]),  # Alpha
                        float(relative_powers[3]),  # Beta1
                        float(relative_powers[4]),  # Beta2
                        float(relative_powers[5]),  # Gamma
                        float(spectral_params['renyi_entropy'][seg_idx]),
                        float(spectral_params['shannon_entropy'][seg_idx]),
                        float(spectral_params['spectral_bandwidth'][seg_idx]),
                        float(spectral_params['spectral_centroid'][seg_idx]),
                        float(spectral_params['spectral_crest_factor'][seg_idx]),
                        float(spectral_params['spectral_edge_frequency_95'][seg_idx]),
                        float(spectral_params['transition_frequency'][seg_idx]),
                        float(spectral_params['tsallis_entropy'][seg_idx]),
                    ]
                    spectral_features_list.append(features)
                    
                    labels_list.append(0 if category == 'HC' else 1)
                    self.sample_to_subject.append(subj_key)

        # Convert to tensors
        self.raw_samples = torch.tensor(np.array(raw_features_list), dtype=torch.float32)
        self.spectral_features = torch.tensor(spectral_features_list, dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels_list), dtype=torch.long)

        # Normalize spectral features
        if normalize_spectral == 'min-max':
            min_vals = self.spectral_features.min(dim=0, keepdim=True).values
            max_vals = self.spectral_features.max(dim=0, keepdim=True).values
            self.spectral_features = (self.spectral_features - min_vals) / (max_vals - min_vals)
        elif normalize_spectral == 'standard':
            mean_vals = self.spectral_features.mean(dim=0, keepdim=True)
            std_vals = self.spectral_features.std(dim=0, keepdim=True)
            self.spectral_features = (self.spectral_features - mean_vals) / std_vals

        logger.debug(f"Raw samples mean: {self.raw_samples.mean()}, std: {self.raw_samples.std()}")
        logger.debug(f"Spectral features mean: {self.spectral_features.mean()}, std: {self.spectral_features.std()}")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x_raw = self.raw_samples[idx]
        x_spectral = self.spectral_features[idx]
        y = self.labels[idx]
        return tuple([x_raw, x_spectral]), y # as a tuple! outside of here, it will just be seen like (x, y)

    def get_sample_to_subject(self, idx: int) -> str:
        return self.sample_to_subject[idx]

