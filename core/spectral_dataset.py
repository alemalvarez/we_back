from torch.utils.data import Dataset
import h5py  # type: ignore
import torch
from typing import Literal, List, Optional
from loguru import logger

class SpectralDataset(Dataset):
    def __init__(self, 
    h5_file_path: str, 
    subjects_txt_path: Optional[str] = None,
    normalize: Literal['min-max', 'standard', 'none'] = 'none',
    subjects_list: Optional[List[str]] = None,
    ):
        if subjects_list:
            logger.info(f"Using provided subjects_list with {len(subjects_list)} subjects")
            self.subject_ids = subjects_list
        else:
            assert subjects_txt_path is not None, "subjects_txt_path must be provided if subjects_list is not provided"
            logger.info(f"Loading features dataset from {h5_file_path} with {len(subjects_txt_path)} subjects")
            with open(subjects_txt_path, 'r') as f:
                self.subject_ids = [line.strip() for line in f.readlines()]

        features_list = []
        labels_list = []
        self.sample_to_subject: List[str] = []

        with h5py.File(h5_file_path, 'r') as f:
            for subj_key in self.subject_ids:
                subject = f['subjects'][subj_key]
                n_segments = subject.attrs['n_segments']
                # Access the spectral parameters group
                spectral_params = subject['spectral/spectral_parameters']
                # For each segment, extract all required features
                for seg_idx in range(n_segments):
                    # Extract relative powers (6 bands)
                    relative_powers = spectral_params['relative_powers'][seg_idx, :]

                    # Create feature vector - expand relative_powers into 6 separate features
                    features = [
                        float(spectral_params['individual_alpha_frequency'][seg_idx]),
                        float(spectral_params['median_frequency'][seg_idx]),
                        # Relative powers for each band:
                        float(relative_powers[0]),  # Delta (0.5-4 Hz)
                        float(relative_powers[1]),  # Theta (4-8 Hz)
                        float(relative_powers[2]),  # Alpha (8-13 Hz)
                        float(relative_powers[3]),  # Beta1 (13-19 Hz)
                        float(relative_powers[4]),  # Beta2 (19-30 Hz)
                        float(relative_powers[5]),  # Gamma (30-70 Hz)
                        float(spectral_params['renyi_entropy'][seg_idx]),
                        float(spectral_params['shannon_entropy'][seg_idx]),
                        float(spectral_params['spectral_bandwidth'][seg_idx]),
                        float(spectral_params['spectral_centroid'][seg_idx]),
                        float(spectral_params['spectral_crest_factor'][seg_idx]),
                        float(spectral_params['spectral_edge_frequency_95'][seg_idx]),
                        float(spectral_params['transition_frequency'][seg_idx]),
                        float(spectral_params['tsallis_entropy'][seg_idx]),
                    ]

                    features_list.append(features)
                    labels_list.append(0 if subject.attrs['category'] == 'HC' else 1)
                    self.sample_to_subject.append(subj_key)

        # Convert to tensors
        self.features = torch.tensor(features_list, dtype=torch.float32)
        self.labels = torch.tensor(labels_list, dtype=torch.float32)

        logger.debug(f"Mean before normalization: {self.features.mean()}")
        logger.debug(f"Std before normalization: {self.features.std()}")

        if normalize == 'min-max':
            min_vals = self.features.min(dim=0, keepdim=True).values
            max_vals = self.features.max(dim=0, keepdim=True).values
            self.features = (self.features - min_vals) / (max_vals - min_vals)
        elif normalize == 'standard':
            mean_vals = self.features.mean(dim=0, keepdim=True)
            std_vals = self.features.std(dim=0, keepdim=True)
            self.features = (self.features - mean_vals) / std_vals

        logger.debug(f"Mean after normalization: {self.features.mean()}")
        logger.debug(f"Std after normalization: {self.features.std()}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_sample_to_subject(self, idx: int) -> str:
        return self.sample_to_subject[idx]

if __name__ == "__main__":
    dataset = SpectralDataset(
        h5_file_path="artifacts/POCTEP_DK_features_only:v0/POCTEP_DK_features_only.h5", 
        subjects_txt_path="experiments/ADSEV_vs_HC/POCTEP/spectral/splits/training_subjects.txt",
        normalize="min-max"
    )
    print(dataset[0])
    print(len(dataset))
    print(dataset.get_sample_to_subject(0))