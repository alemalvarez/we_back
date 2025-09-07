from datetime import datetime
import json
import os
import time
from typing import List

import numpy as np
import h5py
import wandb
import core.eeg_utils as eeg

from loguru import logger

from spectral.median_frequency import calcular_mf_vector
from spectral.spectral_95_limit_frequency import calcular_sef95_vector
from spectral.individual_alpha_frequency_transition_frequency import calcular_iaftf, calcular_iaftf_vector
from spectral.relative_powers import calcular_rp_vector
from spectral.renyi_entropy import calcular_re_vector
from spectral.shannon_entropy import calcular_se_vector
from spectral.tsallis_entropy import calcular_te_vector
from spectral.spectral_crest_factor import calcular_scf_vector
from spectral.spectral_centroid import calcular_sc_vector
from spectral.spectral_bandwidth import calcular_sb_vector
from core.schemas import Subject

INTEREST_BAND: List[float] = [0.5, 70.0]
IAFTF_Q: List[float] = [4.0, 15.0]
NPERSEG: int = 256
Q_RENYI: float = 0.5
Q_TSALLIS: float = 0.5

def create_dataset(
        data_folder: str,
        output_path: str,
        comes_from_bbdds: bool = True
) -> None:
    
    logger.info(f"Creating dataset from {data_folder} to {output_path}")

    try:
        all_files, names = eeg.load_files_from_folder(data_folder)
    except Exception as e:
        logger.error(f"Error loading files from {data_folder}: {e}")
        raise e
    
    if not all_files:
        logger.error(f"No files found in {data_folder}. Exiting.")
        return
    
    subjects: List[Subject] = []
    
    for idx_idx, (mat_file_content, file_name) in enumerate(zip(all_files, names)):
        logger.debug(f"Processing file {idx_idx + 1}/{len(all_files)}: {file_name}")

        try:
            signal_data, cfg, _ = eeg.get_nice_data(
                raw_data=mat_file_content, 
                name=file_name, 
                comes_from_bbdds=comes_from_bbdds
            )

            n_segments, n_samples, n_channels = signal_data.shape
            logger.info(f"Signal segments: {n_segments}, Samples/segment: {n_samples}, Channels: {n_channels}")

            f, pxx_segments = eeg.get_spectral_density(signal_data, cfg, nperseg=NPERSEG)

            if f.size == 0 or pxx_segments.size == 0:
                logger.error(f"PSD calculation resulted in empty arrays for {file_name}. Skipping.")
                continue
            
            logger.debug(f"PSD calculated. Frequencies shape: {f.shape}, Pxx segments shape: {pxx_segments.shape}")

            spectral_params = {
                'median_frequency': calcular_mf_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND),
                'spectral_edge_frequency_95': calcular_sef95_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND),

                'renyi_entropy': calcular_re_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND, q_param=Q_RENYI),
                'shannon_entropy': calcular_se_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND),
                'tsallis_entropy': calcular_te_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND, q_param=Q_TSALLIS),

                'spectral_crest_factor': calcular_scf_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND),

                'relative_powers': calcular_rp_vector(psd=pxx_segments, f=f, banda_total=INTEREST_BAND, sub_bandas=list(eeg.CLASSICAL_BANDS.values()))
            }

            centroids = calcular_sc_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND)
            
            if centroids is not None:
                spectral_bandwith = calcular_sb_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND, spectral_centroids=centroids)
                spectral_params['spectral_bandwidth'] = spectral_bandwith
                spectral_params['spectral_centroid'] = centroids

            individual_alpha_frequency, transition_frequency = calcular_iaftf_vector(psd=pxx_segments, f=f, banda=INTEREST_BAND, q=IAFTF_Q)
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

            subjects.append(Subject(
                category=file_name.split('_')[0],
                file_origin=file_name,
                sampling_rate=cfg['fs'],
                n_segments=n_segments,
                filtering=cfg['filtering'],
                trial_length_secs=cfg['trial_length_secs'],
                ica_components_removed=cfg['N_discarded_ICA'],
                raw_segments=signal_data,
                spectral=Subject.SpectralData(
                    psd=pxx_segments, 
                    f=f, 
                    spectral_parameters=Subject.SpectralData.SpectralParameters(
                        median_frequency=spectral_params['median_frequency'],
                        spectral_edge_frequency_95=spectral_params['spectral_edge_frequency_95'],
                        individual_alpha_frequency=spectral_params['individual_alpha_frequency'],
                        transition_frequency=spectral_params['transition_frequency'],
                        relative_powers=spectral_params['relative_powers'],
                        renyi_entropy=spectral_params['renyi_entropy'],
                        shannon_entropy=spectral_params['shannon_entropy'],
                        tsallis_entropy=spectral_params['tsallis_entropy'],
                        spectral_crest_factor=spectral_params['spectral_crest_factor'],
                        spectral_centroid=spectral_params['spectral_centroid'],
                        spectral_bandwidth=spectral_params['spectral_bandwidth']
                    )
                )
            ))

            logger.info(f"Subject {subjects[-1]}")

        except Exception as e:
            logger.exception(f"Error processing file {file_name}: {e}")
            continue

    logger.info(f"Creating dataset from {data_folder} to {output_path}")

    with h5py.File(output_path, 'w') as f:
        logger.info(f"Creating dataset from {data_folder} to {output_path}")
        f.attrs['description'] = f'Dataset from {data_folder}'
        f.attrs['version'] = '1.0.0'
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['author'] = 'Alejandro'
        f.attrs['license'] = 'MIT'
        f.attrs['source_version'] = '1.0.0'

        f.attrs['classical_bands'] = json.dumps(eeg.CLASSICAL_BANDS)
        f.attrs['interest_band'] = INTEREST_BAND
        f.attrs['iaftf_q'] = IAFTF_Q
        f.attrs['nperseg'] = NPERSEG
        f.attrs['q_renyi'] = Q_RENYI
        f.attrs['q_tsallis'] = Q_TSALLIS

        subjects_group = f.create_group('subjects')

        logger.info(f"Creating dataset from {data_folder} to {output_path}")

        for subject in subjects:
            subj_group = subjects_group.create_group(subject.file_origin)

            subj_group.attrs['category'] = subject.category
            subj_group.attrs['file_origin'] = subject.file_origin
            subj_group.attrs['sampling_rate'] = subject.sampling_rate
            subj_group.attrs['n_segments'] = subject.n_segments
            subj_group.attrs['filtering'] = json.dumps(subject.filtering)
            subj_group.attrs['ica_components_removed'] = subject.ica_components_removed
            subj_group.attrs['trial_length_secs'] = subject.trial_length_secs

            n_samples, n_channels = subject.raw_segments.shape[1:]

            subj_group.create_dataset('raw_segments', data=subject.raw_segments, compression='gzip', chunks=(1, n_samples, n_channels))

            spectral_group = subj_group.create_group('spectral')
            spectral_group.create_dataset('psd', data=subject.spectral.psd, compression='gzip', dtype=np.float32)
            spectral_group.create_dataset('f', data=subject.spectral.f, compression='gzip')

            spectral_params_group = spectral_group.create_group('spectral_parameters')
            spectral_params_group.create_dataset('median_frequency', data=subject.spectral.spectral_parameters.median_frequency, compression='gzip')
            spectral_params_group.create_dataset('spectral_edge_frequency_95', data=subject.spectral.spectral_parameters.spectral_edge_frequency_95, compression='gzip')
            spectral_params_group.create_dataset('individual_alpha_frequency', data=subject.spectral.spectral_parameters.individual_alpha_frequency, compression='gzip')
            spectral_params_group.create_dataset('transition_frequency', data=subject.spectral.spectral_parameters.transition_frequency, compression='gzip')
            spectral_params_group.create_dataset('relative_powers', data=subject.spectral.spectral_parameters.relative_powers, compression='gzip')
            spectral_params_group.create_dataset('renyi_entropy', data=subject.spectral.spectral_parameters.renyi_entropy, compression='gzip')
            spectral_params_group.create_dataset('shannon_entropy', data=subject.spectral.spectral_parameters.shannon_entropy, compression='gzip')
            spectral_params_group.create_dataset('tsallis_entropy', data=subject.spectral.spectral_parameters.tsallis_entropy, compression='gzip')

            spectral_params_group.create_dataset('spectral_crest_factor', data=subject.spectral.spectral_parameters.spectral_crest_factor, compression='gzip')
            spectral_params_group.create_dataset('spectral_centroid', data=subject.spectral.spectral_parameters.spectral_centroid, compression='gzip')
            spectral_params_group.create_dataset('spectral_bandwidth', data=subject.spectral.spectral_parameters.spectral_bandwidth, compression='gzip')

            logger.info(f"Subject {subject.file_origin} written to {output_path}")
                
    
    logger.success(f"Dataset created successfully")


def save_to_wandb(
        dataset_path: str,
        project_name: str = "eeg-efficiency-test"
) -> None:

    logger.info(f"Saving dataset to Weights & Biases")

    with wandb.init(project=project_name, name=f"efficiency-test-{int(time.time())}") as run:

        file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)

        logger.info(f"File size: {file_size_mb} MB")

        dataset = wandb.Artifact(
            "small-test-dataset",
            type="dataset",
            description="Small test dataset",
            metadata={
                "file_size_mb": file_size_mb,
                "source_version": "1.0.0"
            }
        )
        
        dataset.add_file(dataset_path)
        run.log_artifact(dataset)

if __name__ == "__main__":
    create_dataset(
        data_folder="/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH",
        output_path="h5test.h5"
    )

    save_to_wandb(
        dataset_path="h5test.h5"
    )

    