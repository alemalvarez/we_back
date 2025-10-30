from typing import List, Literal, Optional, Dict, Any
import h5py  # type: ignore
from loguru import logger
from torch.utils.data import Dataset
import numpy as np
import torch
from core.schemas import NormalizationStats


class MultiDataset(Dataset):
    """PyTorch dataset combining raw EEG segments and spectral features."""

    raw_samples: torch.Tensor
    spectral_features: torch.Tensor
    labels: torch.Tensor
    sample_to_subject: List[str]
    norm_stats: Optional[NormalizationStats]

    def __init__(
        self, 
        h5_file_path: str, 
        subjects_txt_path: Optional[str] = None,
        normalize_raw: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel-dataset', 'dataset', 'control-channel', 'control-global'] = 'sample-channel',
        normalize_spectral: Literal['min-max', 'standard', 'none'] = 'standard',
        subjects_list: Optional[List[str]] = None,
        norm_stats: Optional[NormalizationStats] = None,
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
                    'folder_id': subject.attrs['folder_id'],
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

        elif normalize_raw == 'channel-dataset':
            # Normalize per channel per database (folder_id)
            raw_stats_dict: Optional[Dict[Any, tuple[float, float]]] = None
            if norm_stats is not None and norm_stats.raw_stats is not None:
                # Use provided stats
                logger.info("Using provided raw normalization stats for channel-dataset")
                provided_stats = norm_stats.raw_stats
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    n_channels = data.shape[2]
                    normalized_segments = data.astype(np.float32).copy()
                    for ch_idx in range(n_channels):
                        key = ('dataset', folder_id, ch_idx)
                        if key in provided_stats:
                            mean_val, std_val = provided_stats[key]
                            normalized_segments[:, :, ch_idx] = (normalized_segments[:, :, ch_idx] - mean_val) / std_val
                        else:
                            logger.warning(f"No stats found for {key}, skipping normalization")
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = norm_stats.raw_stats
            else:
                # Compute stats per database per channel
                logger.info("Computing raw normalization stats for channel-dataset")
                data_by_db_ch: Dict[str, List[np.ndarray]] = {}
                for subj_key in self.subject_ids:
                    folder_id = subject_data[subj_key]['folder_id']
                    if folder_id not in data_by_db_ch:
                        data_by_db_ch[folder_id] = []
                    data_by_db_ch[folder_id].append(subject_data[subj_key]['raw_data'])
                
                stats_dict_ch: Dict[Any, tuple[float, float]] = {}
                for folder_id, db_data_list in data_by_db_ch.items():
                    db_data = np.concatenate(db_data_list, axis=0)
                    n_channels = db_data.shape[2]
                    for ch_idx in range(n_channels):
                        mean_val = db_data[:, :, ch_idx].mean(dtype=np.float32)
                        std_val = db_data[:, :, ch_idx].std(dtype=np.float32)
                        if std_val == 0:
                            std_val = 1.0
                        stats_dict_ch[('dataset', folder_id, ch_idx)] = (float(mean_val), float(std_val))
                
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    n_channels = data.shape[2]
                    normalized_segments = data.astype(np.float32).copy()
                    for ch_idx in range(n_channels):
                        key = ('dataset', folder_id, ch_idx)
                        mean_val, std_val = stats_dict_ch[key]
                        normalized_segments[:, :, ch_idx] = (normalized_segments[:, :, ch_idx] - mean_val) / std_val
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = stats_dict_ch

        elif normalize_raw == 'dataset':
            # Normalize per database (folder_id) globally
            raw_stats_dict: Optional[Dict[Any, tuple[float, float]]] = None
            if norm_stats is not None and norm_stats.raw_stats is not None:
                logger.info("Using provided raw normalization stats for dataset")
                provided_stats = norm_stats.raw_stats
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    key_db = ('dataset', folder_id)
                    if key_db in provided_stats:
                        mean_val, std_val = provided_stats[key_db]
                        normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                    else:
                        logger.warning(f"No stats found for {key_db}, skipping normalization")
                        normalized_segments = data.astype(np.float32)
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = norm_stats.raw_stats
            else:
                logger.info("Computing raw normalization stats for dataset")
                data_by_db_global: Dict[str, List[np.ndarray]] = {}
                for subj_key in self.subject_ids:
                    folder_id = subject_data[subj_key]['folder_id']
                    if folder_id not in data_by_db_global:
                        data_by_db_global[folder_id] = []
                    data_by_db_global[folder_id].append(subject_data[subj_key]['raw_data'])
                
                stats_dict_global: Dict[Any, tuple[float, float]] = {}
                for folder_id, db_data_list in data_by_db_global.items():
                    db_data = np.concatenate(db_data_list, axis=0)
                    mean_val = db_data.mean(dtype=np.float32)
                    std_val = db_data.std(dtype=np.float32)
                    if std_val == 0:
                        std_val = 1.0
                    stats_dict_global[('dataset', folder_id)] = (float(mean_val), float(std_val))
                
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    key_db = ('dataset', folder_id)
                    mean_val, std_val = stats_dict_global[key_db]
                    normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = stats_dict_global

        elif normalize_raw == 'control-channel':
            # Normalize per channel per database using only HC subjects
            raw_stats_dict: Optional[Dict[Any, tuple[float, float]]] = None
            if norm_stats is not None and norm_stats.raw_stats is not None:
                logger.info("Using provided raw normalization stats for control-channel")
                provided_stats = norm_stats.raw_stats
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    n_channels = data.shape[2]
                    normalized_segments = data.astype(np.float32).copy()
                    for ch_idx in range(n_channels):
                        key = ('dataset', folder_id, ch_idx)
                        if key in provided_stats:
                            mean_val, std_val = provided_stats[key]
                            normalized_segments[:, :, ch_idx] = (normalized_segments[:, :, ch_idx] - mean_val) / std_val
                        else:
                            logger.warning(f"No stats found for {key}, skipping normalization")
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = norm_stats.raw_stats
            else:
                logger.info("Computing raw normalization stats for control-channel (HC only)")
                hc_data_by_db_ch: Dict[str, List[np.ndarray]] = {}
                for subj_key in self.subject_ids:
                    if subject_data[subj_key]['category'] == 'HC':
                        folder_id = subject_data[subj_key]['folder_id']
                        if folder_id not in hc_data_by_db_ch:
                            hc_data_by_db_ch[folder_id] = []
                        hc_data_by_db_ch[folder_id].append(subject_data[subj_key]['raw_data'])
                
                stats_dict_ctrl_ch: Dict[Any, tuple[float, float]] = {}
                for folder_id, db_data_list in hc_data_by_db_ch.items():
                    db_data = np.concatenate(db_data_list, axis=0)
                    n_channels = db_data.shape[2]
                    for ch_idx in range(n_channels):
                        mean_val = db_data[:, :, ch_idx].mean(dtype=np.float32)
                        std_val = db_data[:, :, ch_idx].std(dtype=np.float32)
                        if std_val == 0:
                            std_val = 1.0
                        stats_dict_ctrl_ch[('dataset', folder_id, ch_idx)] = (float(mean_val), float(std_val))
                
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    n_channels = data.shape[2]
                    normalized_segments = data.astype(np.float32).copy()
                    for ch_idx in range(n_channels):
                        key = ('dataset', folder_id, ch_idx)
                        if key in stats_dict_ctrl_ch:
                            mean_val, std_val = stats_dict_ctrl_ch[key]
                            normalized_segments[:, :, ch_idx] = (normalized_segments[:, :, ch_idx] - mean_val) / std_val
                        else:
                            logger.warning(f"No HC subjects found for {key}, skipping normalization")
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = stats_dict_ctrl_ch

        elif normalize_raw == 'control-global':
            # Normalize per database using only HC subjects (global across channels)
            raw_stats_dict: Optional[Dict[Any, tuple[float, float]]] = None
            if norm_stats is not None and norm_stats.raw_stats is not None:
                logger.info("Using provided raw normalization stats for control-global")
                provided_stats = norm_stats.raw_stats
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    key_ctrl = ('dataset', folder_id)
                    if key_ctrl in provided_stats:
                        mean_val, std_val = provided_stats[key_ctrl]
                        normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                    else:
                        logger.warning(f"No stats found for {key_ctrl}, skipping normalization")
                        normalized_segments = data.astype(np.float32)
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = norm_stats.raw_stats
            else:
                logger.info("Computing raw normalization stats for control-global (HC only)")
                hc_data_by_db_global: Dict[str, List[np.ndarray]] = {}
                for subj_key in self.subject_ids:
                    if subject_data[subj_key]['category'] == 'HC':
                        folder_id = subject_data[subj_key]['folder_id']
                        if folder_id not in hc_data_by_db_global:
                            hc_data_by_db_global[folder_id] = []
                        hc_data_by_db_global[folder_id].append(subject_data[subj_key]['raw_data'])
                
                stats_dict_ctrl_global: Dict[Any, tuple[float, float]] = {}
                for folder_id, db_data_list in hc_data_by_db_global.items():
                    db_data = np.concatenate(db_data_list, axis=0)
                    mean_val = db_data.mean(dtype=np.float32)
                    std_val = db_data.std(dtype=np.float32)
                    if std_val == 0:
                        std_val = 1.0
                    stats_dict_ctrl_global[('dataset', folder_id)] = (float(mean_val), float(std_val))
                
                normalized_raw_data = []
                for subj_key in self.subject_ids:
                    data = subject_data[subj_key]['raw_data']
                    folder_id = subject_data[subj_key]['folder_id']
                    key_ctrl = ('dataset', folder_id)
                    if key_ctrl in stats_dict_ctrl_global:
                        mean_val, std_val = stats_dict_ctrl_global[key_ctrl]
                        normalized_segments = (data.astype(np.float32) - mean_val) / std_val
                    else:
                        logger.warning(f"No HC subjects found for {key_ctrl}, skipping normalization")
                        normalized_segments = data.astype(np.float32)
                    normalized_raw_data.append(normalized_segments)
                raw_stats_dict = stats_dict_ctrl_global

        else:
            normalized_raw_data = [subject_data[subj_key]['raw_data'] for subj_key in self.subject_ids]
            raw_stats_dict = None
        
        # For normalizations that don't use stats passing, set raw_stats_dict to None
        if normalize_raw in ['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full']:
            raw_stats_dict = None
        
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
        spectral_mean_arr: Optional[np.ndarray] = None
        spectral_std_arr: Optional[np.ndarray] = None
        
        if normalize_spectral == 'min-max':
            if norm_stats is not None and norm_stats.spectral_mean is not None and norm_stats.spectral_std is not None:
                logger.info("Using provided spectral normalization stats for min-max")
                min_vals = torch.from_numpy(norm_stats.spectral_mean)
                max_vals = torch.from_numpy(norm_stats.spectral_std)
                self.spectral_features = (self.spectral_features - min_vals) / (max_vals - min_vals)
                spectral_mean_arr = norm_stats.spectral_mean
                spectral_std_arr = norm_stats.spectral_std
            else:
                min_vals = self.spectral_features.min(dim=0, keepdim=True).values
                max_vals = self.spectral_features.max(dim=0, keepdim=True).values
                self.spectral_features = (self.spectral_features - min_vals) / (max_vals - min_vals)
                spectral_mean_arr = min_vals.numpy()
                spectral_std_arr = max_vals.numpy()
        elif normalize_spectral == 'standard':
            if norm_stats is not None and norm_stats.spectral_mean is not None and norm_stats.spectral_std is not None:
                logger.info("Using provided spectral normalization stats for standard")
                mean_vals = torch.from_numpy(norm_stats.spectral_mean)
                std_vals = torch.from_numpy(norm_stats.spectral_std)
                self.spectral_features = (self.spectral_features - mean_vals) / std_vals
                spectral_mean_arr = norm_stats.spectral_mean
                spectral_std_arr = norm_stats.spectral_std
            else:
                mean_vals = self.spectral_features.mean(dim=0, keepdim=True)
                std_vals = self.spectral_features.std(dim=0, keepdim=True)
                self.spectral_features = (self.spectral_features - mean_vals) / std_vals
                spectral_mean_arr = mean_vals.numpy()
                spectral_std_arr = std_vals.numpy()
        
        # Create combined norm_stats
        self.norm_stats = NormalizationStats(
            raw_stats=raw_stats_dict,
            spectral_mean=spectral_mean_arr,
            spectral_std=spectral_std_arr
        )

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

