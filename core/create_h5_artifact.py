from datetime import datetime
import json
import os
import time
from typing import List, Optional, Set, Union
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

    def create_dataset(
            self,
            data_folders: Union[str, List[str]],
            output_path: Optional[str] = None,
            comes_from_bbdds: bool = True,
            include_raw: bool = True,
            include_psd: bool = True,
            include_features: bool = True,
            dividing_factor: int = 1,
            downsampling_freq: Optional[int] = None
    ) -> None:

        self.dividing_factor = dividing_factor
        self.include_raw = include_raw
        self.include_psd = include_psd
        self.include_features = include_features

        if isinstance(data_folders, str):
            data_folders = [data_folders]

        if output_path is None:
            if len(data_folders) == 1:
                # Single folder: use folder name
                parts = os.path.normpath(data_folders[0]).split(os.sep)
                folder_name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            else:
                # Multiple folders: use combined name
                folder_names = []
                for folder in data_folders:
                    parts = os.path.normpath(folder).split(os.sep)
                    name = parts[-1] if parts else "unknown"
                    folder_names.append(name)
                folder_name = "_".join(folder_names)
            output_path = f"{folder_name}.h5"
            logger.info(f"No output path specified, using default: {output_path}")

        self.dataset_path = output_path

        logger.info(f"Creating dataset from {len(data_folders)} folder(s) to {output_path}")
        for i, folder in enumerate(data_folders, 1):
            logger.info(f"  {i}. {folder}")
        if dividing_factor > 1:
            logger.debug(f"Dividing factor: {dividing_factor}")

        # Load files from all folders
        all_files = []
        names = []
        
        for folder_idx, data_folder in enumerate(data_folders):
            logger.info(f"Loading files from folder {folder_idx + 1}/{len(data_folders)}: {data_folder}")
            try:
                folder_files, folder_names = eeg.load_files_from_folder(data_folder)
                all_files.extend(folder_files)
                names.extend(folder_names)
                logger.info(f"  Found {len(folder_files)} files in {data_folder}")
            except Exception as e:
                logger.error(f"Error loading files from {data_folder}: {e}")
                continue
        
        if not all_files:
            logger.error(f"No files found in any of the {len(data_folders)} folder(s). Exiting.")
            return
            
        logger.info(f"Total files to process: {len(all_files)}")
        
        subjects: List[Subject] = []
        
        seen_fs: Set[int] = set()

        for idx_idx, (mat_file_content, file_name) in enumerate(zip(all_files, names)):
            logger.debug(f"Processing file {idx_idx + 1}/{len(all_files)}: {file_name}")

            try:
                signal_data, cfg, _ = eeg.get_nice_data(
                    raw_data=mat_file_content, 
                    name=file_name, 
                    comes_from_bbdds=comes_from_bbdds
                )

                # Track and validate sampling frequency across files
                fs_in = int(cfg['fs'])
                seen_fs.add(fs_in)
                if downsampling_freq is None and include_raw and len(seen_fs) > 1:
                    raise ValueError(f"Multiple sampling rates found {sorted(seen_fs)} with include_raw=True and no downsampling_freq set.")

                # If requested, resample to target frequency before any further processing
                if downsampling_freq is not None and fs_in != downsampling_freq:
                    logger.info(f"Resampling from {fs_in} Hz to {downsampling_freq} Hz for {file_name}")
                    signal_data = self._resample_segments(signal_data, fs_in=fs_in, fs_out=downsampling_freq)
                    cfg['fs'] = downsampling_freq

                n_segments, n_samples, n_channels = signal_data.shape
                logger.info(f"Signal segments: {n_segments}, Samples/segment: {n_samples}, Channels: {n_channels}")

                # Apply dividing factor if specified
                if dividing_factor > 1:
                    signal_data = self._divide_segments(signal_data, dividing_factor)

                    n_segments, n_samples, n_channels = signal_data.shape
                    logger.info(f"After dividing factor, signal segments: {n_segments}, Samples/segment: {n_samples}, Channels: {n_channels}")

                    
                self.total_segments += n_segments

                # Calculate PSD only if needed
                f, pxx_segments = None, None
                if include_psd:
                    f, pxx_segments = eeg.get_spectral_density(signal_data, cfg, nperseg=self.NPERSEG)

                    if f.size == 0 or pxx_segments.size == 0:
                        logger.error(f"PSD calculation resulted in empty arrays for {file_name}. Skipping.")
                        continue

                    logger.debug(f"PSD calculated. Frequencies shape: {f.shape}, Pxx segments shape: {pxx_segments.shape}")

                # Calculate spectral parameters only if needed
                spectral_params = None
                if include_features:
                    # Need PSD for spectral parameters
                    if f is None or pxx_segments is None:
                        f, pxx_segments = eeg.get_spectral_density(signal_data, cfg, nperseg=self.NPERSEG)

                        if f.size == 0 or pxx_segments.size == 0:
                            logger.error(f"PSD calculation resulted in empty arrays for {file_name}. Skipping.")
                            continue

                        logger.debug(f"PSD calculated for spectral parameters. Frequencies shape: {f.shape}, Pxx segments shape: {pxx_segments.shape}")

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

                    individual_alpha_frequency, transition_frequency = calcular_iaftf_vector(psd=pxx_segments, f=f, banda=self.INTEREST_BAND, q=self.IAFTF_Q)
                    spectral_params['individual_alpha_frequency'] = individual_alpha_frequency
                    spectral_params['transition_frequency'] = transition_frequency

                    logger.info(f"\nSpectral parameters means for {file_name}:")
                    for param_name, param_values in spectral_params.items():
                        if isinstance(param_values, dict):  # Handle relative powers
                            for band_name, band_values in param_values.items():
                                mean_value = np.mean(band_values)
                                logger.info(f"  {param_name} - {band_name}: {mean_value:.4f}")
                        else:
                            mean_value = np.mean(param_values)
                            logger.info(f"  {param_name}: {mean_value:.4f}")

                # Conditionally include data based on flags
                raw_data = signal_data if include_raw else None
                psd_data = pxx_segments if include_psd else None
                f_data = f if include_psd else None

                # Track which features are included
                if include_features and spectral_params is not None:
                    for k, v in spectral_params.items():
                        if v is not None:
                            self.features_included.add(k)

                logger.debug(f"Points per segment: {n_samples}")
                logger.debug(f"Sampling rate: {cfg['fs']}")
                logger.debug(f"Trial length secs: {cfg['trial_length_secs']} (before dividing factor)")

                if dividing_factor > 1:
                    logger.warning(f"Updating trial length secs to {cfg['trial_length_secs'] / dividing_factor}")

                subjects.append(Subject(
                    category=file_name.split('_')[0],
                    file_origin=file_name,
                    sampling_rate=cfg['fs'],
                    n_segments=n_segments,
                    filtering=cfg['filtering'],
                    trial_length_secs=cfg['trial_length_secs'] / dividing_factor,
                    ica_components_removed=cfg['N_discarded_ICA'],
                    points_per_segment=n_samples,
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
                ))

                logger.info(f"Subject {subjects[-1]}")

            except ValueError as e:
                logger.exception(f"Error processing file {file_name}: {e}")
                raise e
            except Exception as e:
                logger.exception(f"Error processing file {file_name}: {e}")
                continue

        self.subjects_saved = len(subjects)

        logger.info(f"Creating dataset from {data_folders} to {output_path}")

        with h5py.File(output_path, 'w') as f:
            assert isinstance(f, h5py.File)

            logger.info(f"Writing H5 file to {output_path}")
            folders_description = "; ".join(data_folders)
            f.attrs['description'] = f'Dataset from {len(data_folders)} folder(s): {folders_description}'
            f.attrs['source_folders'] = json.dumps(data_folders)
            logger.info(f"Creating dataset from {data_folders} to {output_path}")
            f.attrs['version'] = '1.0.0'
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['author'] = 'Alejandro'
            f.attrs['license'] = 'MIT'
            f.attrs['source_version'] = '1.0.0'

            f.attrs['classical_bands'] = json.dumps(eeg.CLASSICAL_BANDS)
            f.attrs['interest_band'] = self.INTEREST_BAND
            f.attrs['iaftf_q'] = self.IAFTF_Q
            f.attrs['nperseg'] = self.NPERSEG
            f.attrs['q_renyi'] = self.Q_RENYI
            f.attrs['q_tsallis'] = self.Q_TSALLIS

            if dividing_factor > 1:
                f.attrs['dividing_factor'] = dividing_factor
            if downsampling_freq is not None:
                f.attrs['fs_downsampled'] = downsampling_freq

            subjects_group = f.create_group('subjects')

            logger.info(f"Creating dataset from {data_folders} to {output_path}")

            for subject in subjects:
                subj_group = subjects_group.create_group(subject.file_origin)

                subj_group.attrs['category'] = subject.category
                subj_group.attrs['file_origin'] = subject.file_origin
                subj_group.attrs['sampling_rate'] = subject.sampling_rate
                subj_group.attrs['n_segments'] = subject.n_segments
                subj_group.attrs['filtering'] = json.dumps(subject.filtering)
                subj_group.attrs['ica_components_removed'] = subject.ica_components_removed
                subj_group.attrs['trial_length_secs'] = subject.trial_length_secs
                subj_group.attrs['points_per_segment'] = subject.points_per_segment

                # Conditionally save raw segments
                if include_raw and subject.raw_segments is not None:
                    n_samples, n_channels = subject.raw_segments.shape[1:]
                    subj_group.create_dataset('raw_segments', data=subject.raw_segments, compression='gzip', chunks=(1, n_samples, n_channels))

                # Create spectral group only if needed
                if subject.spectral is not None:
                    spectral_group = subj_group.create_group('spectral')

                    # Conditionally save PSD data
                    if include_psd and subject.spectral.psd is not None:
                        spectral_group.create_dataset('psd', data=subject.spectral.psd, compression='gzip', dtype=np.float32)
                    if include_psd and subject.spectral.f is not None:
                        spectral_group.create_dataset('f', data=subject.spectral.f, compression='gzip')

                    # Conditionally save spectral parameters
                    if include_features and subject.spectral.spectral_parameters is not None:
                        spectral_params_group = spectral_group.create_group('spectral_parameters')
                        if subject.spectral.spectral_parameters.median_frequency is not None:
                            spectral_params_group.create_dataset('median_frequency', data=subject.spectral.spectral_parameters.median_frequency, compression='gzip')
                        if subject.spectral.spectral_parameters.spectral_edge_frequency_95 is not None:
                            spectral_params_group.create_dataset('spectral_edge_frequency_95', data=subject.spectral.spectral_parameters.spectral_edge_frequency_95, compression='gzip')
                        if subject.spectral.spectral_parameters.individual_alpha_frequency is not None:
                            spectral_params_group.create_dataset('individual_alpha_frequency', data=subject.spectral.spectral_parameters.individual_alpha_frequency, compression='gzip')
                        if subject.spectral.spectral_parameters.transition_frequency is not None:
                            spectral_params_group.create_dataset('transition_frequency', data=subject.spectral.spectral_parameters.transition_frequency, compression='gzip')
                        if subject.spectral.spectral_parameters.relative_powers is not None:
                            spectral_params_group.create_dataset('relative_powers', data=subject.spectral.spectral_parameters.relative_powers, compression='gzip')
                        if subject.spectral.spectral_parameters.renyi_entropy is not None:
                            spectral_params_group.create_dataset('renyi_entropy', data=subject.spectral.spectral_parameters.renyi_entropy, compression='gzip')
                        if subject.spectral.spectral_parameters.shannon_entropy is not None:
                            spectral_params_group.create_dataset('shannon_entropy', data=subject.spectral.spectral_parameters.shannon_entropy, compression='gzip')
                        if subject.spectral.spectral_parameters.tsallis_entropy is not None:
                            spectral_params_group.create_dataset('tsallis_entropy', data=subject.spectral.spectral_parameters.tsallis_entropy, compression='gzip')
                        if subject.spectral.spectral_parameters.spectral_crest_factor is not None:
                            spectral_params_group.create_dataset('spectral_crest_factor', data=subject.spectral.spectral_parameters.spectral_crest_factor, compression='gzip')
                        if subject.spectral.spectral_parameters.spectral_centroid is not None:
                            spectral_params_group.create_dataset('spectral_centroid', data=subject.spectral.spectral_parameters.spectral_centroid, compression='gzip')
                        if subject.spectral.spectral_parameters.spectral_bandwidth is not None:
                            spectral_params_group.create_dataset('spectral_bandwidth', data=subject.spectral.spectral_parameters.spectral_bandwidth, compression='gzip')

                logger.info(f"Subject {subject.file_origin} written to {output_path}")
                    
        
        logger.success("Dataset created successfully")

        # Summary statistics
        logger.info("Summary statistics:")
        logger.info(f"  Subjects saved: {self.subjects_saved}")
        logger.info(f"  Total segments: {self.total_segments}")
        if include_features and self.features_included:
            logger.info(f"  Features included: {sorted(self.features_included)}")
        else:
            logger.info("  Features included: None")

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
    # Example with default output path (will use HURH.h5)
    # create_dataset(
    #     data_folder="/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH",
    #     comes_from_bbdds=False
    # )

    # Example with custom flags for smaller dataset
    creator = DatasetCreator()

    creator.create_dataset(
        data_folders=[
            "/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/POCTEP",
            "/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH"
        ],
        output_path="h5test_raw_only_downsampled.h5",
        include_raw=True,
        include_psd=False,
        include_features=True,
        dividing_factor=1,
        downsampling_freq=200

    )

    # save_to_wandb(
    #     dataset_path="HURH.h5"
    # )
