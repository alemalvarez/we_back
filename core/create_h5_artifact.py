from datetime import datetime
import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import math

import numpy as np
import h5py  # type: ignore
from scipy.signal import resample_poly  # type: ignore
import wandb
import core.eeg_utils as eeg

from loguru import logger

from spectral.median_frequency import calcular_mf_vector
from spectral.spectral_95_limit_frequency import calcular_sef95_vector
from spectral.individual_alpha_frequency_transition_frequency import calcular_iaftf_vector
from spectral.relative_powers import calcular_rp_vector
from spectral.renyi_entropy import calcular_re_vector
from spectral.shannon_entropy import calcular_se_vector
from spectral.tsallis_entropy import calcular_te_vector
from spectral.spectral_crest_factor import calcular_scf_vector
from spectral.spectral_centroid import calcular_sc_vector
from spectral.spectral_bandwidth import calcular_sb_vector
from core.schemas import Subject


class DataSourceConfig:
    """Configuration for a data source folder."""
    def __init__(self, path: str, is_eeg: bool = True):
        self.path = path
        self.is_eeg = is_eeg
        # Extract folder identifier (parent of DK folder, or last two path components)
        parts = os.path.normpath(path).split(os.sep)
        self.folder_id = parts[-2] if len(parts) >= 2 and parts[-1].upper() == 'DK' else parts[-1]


class DatasetCreator:
    """Creates H5 datasets with tracked metadata for W&B artifact uploads."""
    
    def __init__(self):
        self._reset_metadata()
    
    def _reset_metadata(self) -> None:
        """Reset all tracked metadata."""
        self.dividing_factor: int = 2
        self.include_raw: bool = False
        self.include_psd: bool = False
        self.include_features: bool = True
        self.subjects_saved: int = 0
        self.total_segments: int = 0
        self.category_counts: Dict[str, int] = {}
        self.features_included: Set[str] = {
            'individual_alpha_frequency',
            'median_frequency',
            'relative_powers',
            'renyi_entropy',
            'shannon_entropy',
            'spectral_bandwidth',
            'spectral_centroid',
            'spectral_crest_factor',
            'spectral_edge_frequency_95',
            'transition_frequency',
            'tsallis_entropy'
        }
        self.dataset_path: Optional[str] = None
    
    def _divide_segments(self, signal_data: np.ndarray, dividing_factor: int) -> np.ndarray:
        """
        Divide EEG segments into smaller segments by splitting along the time axis.
        
        Args:
            signal_data: EEG signal with shape (n_segments, n_samples, n_channels)
            dividing_factor: Factor by which to divide each segment
        
        Returns:
            np.ndarray: Divided signal with shape (n_segments * dividing_factor, n_samples // dividing_factor, n_channels)
        """
        if dividing_factor <= 1:
            return signal_data
        
        n_segments, n_samples, n_channels = signal_data.shape
        
        # Check if n_samples is divisible by dividing_factor
        if n_samples % dividing_factor != 0:
            # Trim to make it divisible
            new_n_samples = (n_samples // dividing_factor) * dividing_factor
            signal_data = signal_data[:, :new_n_samples, :]
            n_samples = new_n_samples
            logger.warning(f"Trimmed samples from {signal_data.shape[1]} to {n_samples} to make divisible by {dividing_factor}")
    
        samples_per_segment = n_samples // dividing_factor
        
        # Reshape to split segments
        reshaped = signal_data.reshape(n_segments, dividing_factor, samples_per_segment, n_channels)
        # Reorder dimensions to get (n_segments * dividing_factor, samples_per_segment, n_channels)
        divided_signal = reshaped.transpose(0, 1, 2, 3).reshape(n_segments * dividing_factor, samples_per_segment, n_channels)
        
        return divided_signal


    def _resample_segments(self, signal_data: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
        """Anti-aliased resampling of segments along the time axis.

        Args:
            signal_data: Array with shape (n_segments, n_samples, n_channels)
            fs_in: Original sampling rate
            fs_out: Target sampling rate

        Returns:
            np.ndarray: Resampled array with shape (n_segments, new_samples, n_channels)
        """
        if fs_in == fs_out:
            return signal_data

        up = fs_out
        down = fs_in
        g = math.gcd(up, down)
        up //= g
        down //= g

        # Resample along the samples axis (axis=1)
        return resample_poly(signal_data, up, down, axis=1)

    INTEREST_BAND: List[float] = [0.5, 70.0]
    IAFTF_Q: List[float] = [4.0, 15.0]
    NPERSEG: int = 256
    Q_RENYI: float = 0.5
    Q_TSALLIS: float = 0.5

    @staticmethod
    def _extract_category_from_filename(filename: str) -> Optional[str]:
        """Extract category from filename (format: CATEGORY_ID.mat).
        
        Returns:
            Category string or None if format doesn't match
        """
        # Remove .mat extension if present
        base_name = filename.replace('.mat', '')
        
        # Split by underscore and take first part
        parts = base_name.split('_')
        if len(parts) >= 2:
            return parts[0]
        return None

    def _load_files_from_folders(
        self, 
        data_sources: List[DataSourceConfig],
        category_whitelist: Optional[Set[str]] = None
    ) -> List[Tuple[Dict, Dict[str, Union[str, bool]]]]:
        """Load files from all configured data sources.
        
        Args:
            data_sources: List of data source configurations
            category_whitelist: Optional set of allowed categories. If provided, 
                               files with categories not in this set will be ignored.
        
        Returns:
            List of tuples: (file_content, metadata_dict)
            where metadata_dict contains: name, source_folder, folder_id, is_eeg
        """
        all_file_data: List[Tuple[Dict, Dict[str, Union[str, bool]]]] = []
        
        for source in data_sources:
            logger.info(f"Loading files from: {source.path} (is_eeg={source.is_eeg})")
            try:
                folder_files, folder_names = eeg.load_files_from_folder(source.path)
                
                loaded_count = 0
                ignored_count = 0
                
                for file_content, file_name in zip(folder_files, folder_names):
                    # Check category whitelist if provided
                    if category_whitelist is not None:
                        category = self._extract_category_from_filename(file_name)
                        if category is None or category not in category_whitelist:
                            ignored_count += 1
                            continue
                    
                    metadata: Dict[str, Union[str, bool]] = {
                        'name': file_name,
                        'source_folder': source.path,
                        'folder_id': source.folder_id,
                        'is_eeg': source.is_eeg
                    }
                    all_file_data.append((file_content, metadata))
                    loaded_count += 1
                
                logger.info(f"  Loaded {loaded_count} subjects, ignored {ignored_count}")
            except Exception as e:
                logger.error(f"Error loading files from {source.path}: {e}")
                continue
        
        return all_file_data

    def _process_subject_file(
        self,
        mat_file_content: Dict,
        metadata: Dict[str, Union[str, bool]],
        downsampling_freq: Optional[int],
        seen_fs: Set[int]
    ) -> Optional[Dict]:
        """Process a single subject file.
        
        Returns:
            Dict with processed data or None if processing failed
        """
        file_name = str(metadata['name'])
        is_eeg = bool(metadata['is_eeg'])
        
        try:
            signal_data, cfg, _ = eeg.get_nice_data(
                raw_data=mat_file_content, 
                name=file_name, 
                comes_from_bbdds=is_eeg
            )

            # Track and validate sampling frequency
            fs_in = int(cfg['fs'])
            seen_fs.add(fs_in)
            if downsampling_freq is None and len(seen_fs) > 1:
                raise ValueError(
                    f"Multiple sampling rates found {sorted(seen_fs)} and no downsampling_freq set. "
                    f"This will cause inconsistent spectral feature resolution."
                )

            # Resample if needed
            if downsampling_freq is not None and fs_in != downsampling_freq:
                logger.info(f"Resampling from {fs_in} Hz to {downsampling_freq} Hz for {file_name}")
                signal_data = self._resample_segments(signal_data, fs_in=fs_in, fs_out=downsampling_freq)
                cfg['fs'] = downsampling_freq

            n_segments, n_samples, n_channels = signal_data.shape
            logger.info(f"Signal segments: {n_segments}, Samples/segment: {n_samples}, Channels: {n_channels}")

            # Apply dividing factor
            if self.dividing_factor > 1:
                signal_data = self._divide_segments(signal_data, self.dividing_factor)
                n_segments, n_samples, n_channels = signal_data.shape
                logger.info(f"After dividing factor, signal segments: {n_segments}, Samples/segment: {n_samples}, Channels: {n_channels}")

            self.total_segments += n_segments

            # Calculate PSD if needed
            f, pxx_segments = None, None
            if self.include_psd or self.include_features:
                f, pxx_segments = eeg.get_spectral_density(signal_data, cfg, nperseg=self.NPERSEG)

                if f.size == 0 or pxx_segments.size == 0:
                    logger.error(f"PSD calculation resulted in empty arrays for {file_name}. Skipping.")
                    return None

                logger.debug(f"PSD calculated. Frequencies shape: {f.shape}, Pxx segments shape: {pxx_segments.shape}")

            # Calculate spectral features if needed
            spectral_params = None
            if self.include_features and f is not None and pxx_segments is not None:
                spectral_params = self._calculate_spectral_features(f, pxx_segments, file_name)

            return {
                'signal_data': signal_data,
                'cfg': cfg,
                'n_segments': n_segments,
                'n_samples': n_samples,
                'f': f,
                'pxx_segments': pxx_segments,
                'spectral_params': spectral_params,
                'metadata': metadata
            }

        except ValueError as e:
            logger.exception(f"Error processing file {file_name}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Error processing file {file_name}: {e}")
            return None

    def _calculate_spectral_features(
        self,
        f: np.ndarray,
        pxx_segments: np.ndarray,
        file_name: str
    ) -> Dict[str, np.ndarray]:
        """Calculate spectral features from PSD."""
        spectral_params = {
            'median_frequency': calcular_mf_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND),
            'spectral_edge_frequency_95': calcular_sef95_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND),
            'renyi_entropy': calcular_re_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND, q_param=self.Q_RENYI),
            'shannon_entropy': calcular_se_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND),
            'tsallis_entropy': calcular_te_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND, q_param=self.Q_TSALLIS),
            'spectral_crest_factor': calcular_scf_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND),
            'relative_powers': calcular_rp_vector(psd=pxx_segments, f=f, banda_total=self.INTEREST_BAND, sub_bandas=list(eeg.CLASSICAL_BANDS.values()))
        }

        centroids = calcular_sc_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND)
        if centroids is not None:
            spectral_bandwith = calcular_sb_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND, spectral_centroids=centroids)
            spectral_params['spectral_bandwidth'] = spectral_bandwith
            spectral_params['spectral_centroid'] = centroids

        individual_alpha_frequency, transition_frequency = calcular_iaftf_vector(
            psd=pxx_segments, f=f, banda=self.INTEREST_BAND, q=self.IAFTF_Q
        )
        spectral_params['individual_alpha_frequency'] = individual_alpha_frequency
        spectral_params['transition_frequency'] = transition_frequency

        logger.info(f"\nSpectral parameters means for {file_name}:")
        for param_name, param_values in spectral_params.items():
            if isinstance(param_values, dict):
                for band_name, band_values in param_values.items():
                    mean_value = np.mean(band_values)
                    logger.info(f"  {param_name} - {band_name}: {mean_value:.4f}")
            else:
                mean_value = np.mean(param_values)
                logger.info(f"  {param_name}: {mean_value:.4f}")

        return spectral_params

    def _create_subject_object(self, processed_data: Dict) -> Subject:
        """Create a Subject object from processed data."""
        metadata = processed_data['metadata']
        cfg = processed_data['cfg']
        
        raw_data = processed_data['signal_data'] if self.include_raw else None
        psd_data = processed_data['pxx_segments'] if self.include_psd else None
        f_data = processed_data['f'] if self.include_psd else None
        spectral_params = processed_data['spectral_params']

        # Track which features are included
        if self.include_features and spectral_params is not None:
            for k, v in spectral_params.items():
                if v is not None:
                    self.features_included.add(k)

        logger.debug(f"Points per segment: {processed_data['n_samples']}")
        logger.debug(f"Sampling rate: {cfg['fs']}")
        logger.debug(f"Trial length secs: {cfg['trial_length_secs']} (before dividing factor)")

        if self.dividing_factor > 1:
            logger.debug(f"Updating trial length secs to {cfg['trial_length_secs'] / self.dividing_factor}")

        file_name = str(metadata['name'])
        category = file_name.split('_')[0]
        
        # Track category counts
        self.category_counts[category] = self.category_counts.get(category, 0) + 1
        
        return Subject(
            category=category,
            file_origin=file_name,
            source_folder=str(metadata['source_folder']),
            folder_id=str(metadata['folder_id']),
            is_eeg=bool(metadata['is_eeg']),
            sampling_rate=cfg['fs'],
            n_segments=processed_data['n_segments'],
            filtering=cfg['filtering'],
            trial_length_secs=cfg['trial_length_secs'] / self.dividing_factor,
            ica_components_removed=cfg['N_discarded_ICA'],
            points_per_segment=processed_data['n_samples'],
            raw_segments=raw_data,
            spectral=Subject.SpectralData(
                psd=psd_data,
                f=f_data,
                spectral_parameters=Subject.SpectralData.SpectralParameters(
                    median_frequency=spectral_params['median_frequency'] if spectral_params else None,
                    spectral_edge_frequency_95=spectral_params['spectral_edge_frequency_95'] if spectral_params else None,
                    individual_alpha_frequency=spectral_params['individual_alpha_frequency'] if spectral_params else None,
                    transition_frequency=spectral_params['transition_frequency'] if spectral_params else None,
                    relative_powers=spectral_params['relative_powers'] if spectral_params else None,
                    renyi_entropy=spectral_params['renyi_entropy'] if spectral_params else None,
                    shannon_entropy=spectral_params['shannon_entropy'] if spectral_params else None,
                    tsallis_entropy=spectral_params['tsallis_entropy'] if spectral_params else None,
                    spectral_crest_factor=spectral_params['spectral_crest_factor'] if spectral_params else None,
                    spectral_centroid=spectral_params['spectral_centroid'] if spectral_params else None,
                    spectral_bandwidth=spectral_params['spectral_bandwidth'] if spectral_params else None
                ) if spectral_params else None
            ) if (psd_data is not None or f_data is not None or spectral_params is not None) else None
        )

    def _write_h5_file(self, output_path: str, subjects: List[Subject], data_sources: List[DataSourceConfig]) -> None:
        """Write subjects to H5 file."""
        with h5py.File(output_path, 'w') as f:
            assert isinstance(f, h5py.File)

            logger.info(f"Writing H5 file to {output_path}")
            
            # File metadata
            folders_description = "; ".join([s.path for s in data_sources])
            f.attrs['description'] = f'Dataset from {len(data_sources)} folder(s): {folders_description}'
            f.attrs['source_folders'] = json.dumps([s.path for s in data_sources])
            f.attrs['version'] = '1.0.0'
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['author'] = 'Alejandro'
            f.attrs['license'] = 'MIT'
            f.attrs['source_version'] = '1.0.0'

            # Spectral configuration
            f.attrs['classical_bands'] = json.dumps(eeg.CLASSICAL_BANDS)
            f.attrs['interest_band'] = self.INTEREST_BAND
            f.attrs['iaftf_q'] = self.IAFTF_Q
            f.attrs['nperseg'] = self.NPERSEG
            f.attrs['q_renyi'] = self.Q_RENYI
            f.attrs['q_tsallis'] = self.Q_TSALLIS

            if self.dividing_factor > 1:
                f.attrs['dividing_factor'] = self.dividing_factor

            subjects_group = f.create_group('subjects')

            # Track subject names to handle duplicates
            name_counts: Dict[str, int] = {}

            for subject in subjects:
                # Create disambiguated name
                base_name = f"{subject.folder_id}_{subject.file_origin}"
                
                # Handle duplicates by appending counter
                if base_name in name_counts:
                    name_counts[base_name] += 1
                    disambiguated_name = f"{base_name}_{name_counts[base_name]}"
                else:
                    name_counts[base_name] = 0
                    disambiguated_name = base_name

                subj_group = subjects_group.create_group(disambiguated_name)

                # Subject metadata
                subj_group.attrs['category'] = subject.category
                subj_group.attrs['file_origin'] = subject.file_origin
                subj_group.attrs['source_folder'] = subject.source_folder
                subj_group.attrs['folder_id'] = subject.folder_id
                subj_group.attrs['is_eeg'] = subject.is_eeg
                subj_group.attrs['sampling_rate'] = subject.sampling_rate
                subj_group.attrs['n_segments'] = subject.n_segments
                subj_group.attrs['filtering'] = json.dumps(subject.filtering)
                subj_group.attrs['ica_components_removed'] = subject.ica_components_removed
                subj_group.attrs['trial_length_secs'] = subject.trial_length_secs
                subj_group.attrs['points_per_segment'] = subject.points_per_segment

                # Raw segments
                if self.include_raw and subject.raw_segments is not None:
                    n_samples, n_channels = subject.raw_segments.shape[1:]
                    subj_group.create_dataset(
                        'raw_segments', 
                        data=subject.raw_segments, 
                        compression='gzip', 
                        chunks=(1, n_samples, n_channels)
                    )

                # Spectral data
                if subject.spectral is not None:
                    spectral_group = subj_group.create_group('spectral')

                    if self.include_psd and subject.spectral.psd is not None:
                        spectral_group.create_dataset('psd', data=subject.spectral.psd, compression='gzip', dtype=np.float32)
                    if self.include_psd and subject.spectral.f is not None:
                        spectral_group.create_dataset('f', data=subject.spectral.f, compression='gzip')

                    if self.include_features and subject.spectral.spectral_parameters is not None:
                        spectral_params_group = spectral_group.create_group('spectral_parameters')
                        params = subject.spectral.spectral_parameters
                        
                        if params.median_frequency is not None:
                            spectral_params_group.create_dataset('median_frequency', data=params.median_frequency, compression='gzip')
                        if params.spectral_edge_frequency_95 is not None:
                            spectral_params_group.create_dataset('spectral_edge_frequency_95', data=params.spectral_edge_frequency_95, compression='gzip')
                        if params.individual_alpha_frequency is not None:
                            spectral_params_group.create_dataset('individual_alpha_frequency', data=params.individual_alpha_frequency, compression='gzip')
                        if params.transition_frequency is not None:
                            spectral_params_group.create_dataset('transition_frequency', data=params.transition_frequency, compression='gzip')
                        if params.relative_powers is not None:
                            spectral_params_group.create_dataset('relative_powers', data=params.relative_powers, compression='gzip')
                        if params.renyi_entropy is not None:
                            spectral_params_group.create_dataset('renyi_entropy', data=params.renyi_entropy, compression='gzip')
                        if params.shannon_entropy is not None:
                            spectral_params_group.create_dataset('shannon_entropy', data=params.shannon_entropy, compression='gzip')
                        if params.tsallis_entropy is not None:
                            spectral_params_group.create_dataset('tsallis_entropy', data=params.tsallis_entropy, compression='gzip')
                        if params.spectral_crest_factor is not None:
                            spectral_params_group.create_dataset('spectral_crest_factor', data=params.spectral_crest_factor, compression='gzip')
                        if params.spectral_centroid is not None:
                            spectral_params_group.create_dataset('spectral_centroid', data=params.spectral_centroid, compression='gzip')
                        if params.spectral_bandwidth is not None:
                            spectral_params_group.create_dataset('spectral_bandwidth', data=params.spectral_bandwidth, compression='gzip')

                logger.info(f"Subject {disambiguated_name} written to {output_path}")

    def create_dataset(
            self,
            data_sources: Union[str, List[str], List[Dict[str, Union[str, object]]]],
            output_path: Optional[str] = None,
            is_eeg: bool = True,
            include_raw: bool = True,
            include_psd: bool = True,
            include_features: bool = True,
            dividing_factor: int = 1,
            downsampling_freq: Optional[int] = None,
            category_whitelist: Optional[List[str]] = None
    ) -> None:
        """Create H5 dataset from multiple data sources.
        
        Args:
            data_sources: Can be:
                - A single folder path (str)
                - List of folder paths (List[str])
                - List of dicts with {"path": str, "is_eeg": bool}
            output_path: Optional output path for H5 file
            is_eeg: Default is_eeg flag (used if data_sources is str or List[str])
            include_raw: Include raw signal data
            include_psd: Include PSD data
            include_features: Include spectral features
            dividing_factor: Factor to divide segments by
            downsampling_freq: Optional target sampling frequency
            category_whitelist: Optional list of allowed categories. 
                               Defaults to ['ADSEV', 'ADMIL', 'HC', 'MCI', 'AD', 'ADMOD']
        """
        self.dividing_factor = dividing_factor
        self.include_raw = include_raw
        self.include_psd = include_psd
        self.include_features = include_features
        
        # Set default category whitelist if not provided
        if category_whitelist is None:
            category_whitelist = ['ADSEV', 'ADMIL', 'HC', 'MCI', 'AD', 'ADMOD']
        
        whitelist_set: Optional[Set[str]] = set(category_whitelist) if category_whitelist else None
        
        if whitelist_set:
            logger.info(f"Category whitelist enabled: {sorted(whitelist_set)}")

        # Normalize data_sources to List[DataSourceConfig]
        sources: List[DataSourceConfig] = []
        if isinstance(data_sources, str):
            sources = [DataSourceConfig(data_sources, is_eeg)]
        elif isinstance(data_sources, list):
            if len(data_sources) > 0 and all(isinstance(s, str) for s in data_sources):
                sources = [DataSourceConfig(str(path), is_eeg) for path in data_sources]
            elif len(data_sources) > 0 and all(isinstance(s, dict) for s in data_sources):
                for source_dict in data_sources:
                    if not isinstance(source_dict, dict):
                        raise ValueError("Expected dict in data_sources")
                    path_val = source_dict.get('path')  # type: ignore
                    is_eeg_val = source_dict.get('is_eeg', True)  # type: ignore
                    if not isinstance(path_val, str):
                        raise ValueError("path must be a string")
                    sources.append(DataSourceConfig(path_val, bool(is_eeg_val)))
            else:
                raise ValueError("data_sources must be all strings or all dicts")
        else:
            raise ValueError("data_sources must be str, List[str], or List[dict]")

        # Generate default output path if needed
        if output_path is None:
            if len(sources) == 1:
                parts = os.path.normpath(sources[0].path).split(os.sep)
                folder_name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            else:
                folder_name = "_".join([s.folder_id for s in sources])
            output_path = f"{folder_name}.h5"
            logger.info(f"No output path specified, using default: {output_path}")

        self.dataset_path = output_path

        logger.info(f"Creating dataset from {len(sources)} source(s) to {output_path}")
        for i, source in enumerate(sources, 1):
            logger.info(f"  {i}. {source.path} (is_eeg={source.is_eeg})")
        if dividing_factor > 1:
            logger.debug(f"Dividing factor: {dividing_factor}")

        # Load files from all sources
        all_file_data = self._load_files_from_folders(sources, whitelist_set)
        
        if not all_file_data:
            logger.error(f"No files found in any of the {len(sources)} source(s). Exiting.")
            return
            
        logger.info(f"Total files to process: {len(all_file_data)}")
        
        # Process all files
        subjects: List[Subject] = []
        seen_fs: Set[int] = set()

        for idx, (mat_file_content, metadata) in enumerate(all_file_data, 1):
            logger.debug(f"Processing file {idx}/{len(all_file_data)}: {metadata['name']}")
            
            processed_data = self._process_subject_file(
                mat_file_content, 
                metadata, 
                downsampling_freq, 
                seen_fs
            )
            
            if processed_data is None:
                continue
            
            subject = self._create_subject_object(processed_data)
            subjects.append(subject)
            logger.info(f"Subject {subject.file_origin} processed")

        self.subjects_saved = len(subjects)

        # Write H5 file
        self._write_h5_file(output_path, subjects, sources)
        
        # Save downsampling freq to H5 metadata if used
        if downsampling_freq is not None:
            with h5py.File(output_path, 'a') as f:
                f.attrs['fs_downsampled'] = downsampling_freq
        
        logger.success("Dataset created successfully")

        # Summary statistics
        logger.info("Summary statistics:")
        logger.info(f"  Subjects saved: {self.subjects_saved}")
        logger.info(f"  Total segments: {self.total_segments}")
        
        # Print category counts
        if self.category_counts:
            logger.info("\n  Subjects per category:")
            for category in sorted(self.category_counts.keys()):
                count = self.category_counts[category]
                logger.info(f"    {category}: {count}")
        
        if include_features and self.features_included:
            logger.info(f"\n  Features included: {sorted(self.features_included)}")
        else:
            logger.info("\n  Features included: None")

    def save_to_wandb(
            self,
            project_name: str = "dataset-creation"
    ) -> None:
        if self.dataset_path is None:
            logger.error("No dataset path available. Create a dataset first.")
            return

        logger.info("Saving dataset to Weights & Biases")

        with wandb.init(project=project_name, name=f"efficiency-test-{int(time.time())}") as run:

            file_size_mb = os.path.getsize(self.dataset_path) / (1024 * 1024)

            logger.info(f"File size: {file_size_mb} MB")
            filename = os.path.basename(self.dataset_path)

            dataset = wandb.Artifact(
                    filename,
                    type="dataset",
                    description=f"Dataset from {self.dataset_path}",
                    metadata={
                        "file_size_mb": file_size_mb,
                        "source_version": "1.0.0",
                        "dividing_factor": self.dividing_factor,
                        "contains_raw": self.include_raw,
                        "contains_psd": self.include_psd,
                        "contains_features": self.include_features,
                        "subjects_saved": self.subjects_saved,
                        "total_segments": self.total_segments,
                        "features": sorted(self.features_included) if self.features_included else [],
                    }
                )
            
            dataset.add_file(self.dataset_path)
            run.log_artifact(dataset)

if __name__ == "__main__":
    creator = DatasetCreator()

    # Example 1: Simple list of folders (all with same is_eeg setting)
    # creator.create_dataset(
    #     data_sources=[
    #         "/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/POCTEP",
    #         "/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH"
    #     ],
    #     output_path="testtest.h5",
    #     is_eeg=True,
    #     include_raw=True,
    #     include_psd=True,
    #     include_features=True,
    #     dividing_factor=1,
    #     downsampling_freq=200
    # )

    # Example 2: Mixed sources with per-folder is_eeg configuration
    creator.create_dataset(
        data_sources=[
            {"path": "/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/POCTEP", "is_eeg": True},
            {"path": "/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH", "is_eeg": True},
            {"path": "/Users/alemalvarez/code-workspace/TFG/DATA/MEG/DK", "is_eeg": False}
        ],
        output_path="megatest.h5",
        include_raw=True,
        include_psd=True,
        include_features=True,
        downsampling_freq=200,
        category_whitelist = ['ADSEV', 'ADMIL', 'HC', 'AD', 'ADMOD']
    )
    # creator.create_dataset(
    #     data_sources=[
    #         # eegs
    #         {"path": r"E:\BBDDs\HURH\DK", "is_eeg": True},
    #         {"path": r"E:\BBDDs\POCTEP\DK", "is_eeg": True},
    #         # megs
    #         {"path": r"E:\BBDDs\MEG\BrainDock\DK", "is_eeg": False},
    #         {"path": r"E:\BBDDs\MEG\Control\DK", "is_eeg": False},
    #         {"path": r"E:\BBDDs\MEG\Kakehashi\DK", "is_eeg": False},
    #         {"path": r"E:\BBDDs\MEG\Kumagaya General\DK", "is_eeg": False},
    #         {"path": r"E:\BBDDs\MEG\Kumagaya Monowasure\DK", "is_eeg": False},
    #         {"path": r"E:\BBDDs\MEG\Mihara\DK", "is_eeg": False},
    #         {"path": r"E:\BBDDs\MEG\Monowasure/\DK", "is_eeg": False}

    #     ],
    #     output_path="megatest.h5",
    #     include_raw=True,
    #     include_psd=True,
    #     include_features=True,
    #     downsampling_freq=200,
    #     category_whitelist = ['ADSEV', 'ADMIL', 'HC', 'AD', 'ADMOD', 'MCI']
    # )
