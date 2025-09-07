from dataclasses import dataclass
from datetime import datetime
import json
import h5py # type: ignore
import numpy as np

from loguru import logger

@dataclass 
class Subject:
    @dataclass
    class SpectralData:

        @dataclass
        class SpectralParameters:
            median_frequency: np.ndarray
            spectral_edge_frequency_95: np.ndarray
            individual_alpha_frequency: np.ndarray
            transition_frequency: np.ndarray
            relative_powers: dict[str, np.ndarray]  # Each array has shape (n_segments,)
            renyi_entropy: np.ndarray
            shannon_entropy: np.ndarray
            tsallis_entropy: np.ndarray
            spectral_crest_factor: np.ndarray
            spectral_centroid: np.ndarray
            
        psd: np.ndarray
        f: np.ndarray
        spectral_parameters: SpectralParameters

    # Identifying information
    category: str
    file_origin: str

    # Data information
    sampling_rate: int
    n_segments: int
    filtering: dict
    ica_components_removed: int

    raw_segments: np.ndarray
    spectral: SpectralData

    def __str__(self) -> str:
        """String representation of the Subject class."""
        return (
            f"Subject(\n"
            f"  category: {self.category}\n"
            f"  file_origin: {self.file_origin}\n"
            f"  sampling_rate: {self.sampling_rate} Hz\n"
            f"  n_segments: {self.n_segments}\n"
            f"  filtering: {self.filtering}\n"
            f"  ica_components_removed: {self.ica_components_removed}\n"
            f"  raw_segments: shape {self.raw_segments.shape}\n"
            f"  spectral: SpectralData(\n"
            f"    psd: shape {self.spectral.psd.shape}\n"
            f"    f: shape {self.spectral.f.shape}\n"
            f"    spectral_parameters: SpectralParameters(\n"
            f"      median_frequency: shape {self.spectral.spectral_parameters.median_frequency.shape}\n"
            f"      spectral_edge_frequency_95: shape {self.spectral.spectral_parameters.spectral_edge_frequency_95.shape}\n"
            f"      individual_alpha_frequency: shape {self.spectral.spectral_parameters.individual_alpha_frequency.shape}\n"
            f"      transition_frequency: shape {self.spectral.spectral_parameters.transition_frequency.shape}\n"
            f"      relative_powers: {', '.join(f'{k}: shape {v.shape}' for k, v in self.spectral.spectral_parameters.relative_powers.items())}\n"
            f"      renyi_entropy: shape {self.spectral.spectral_parameters.renyi_entropy.shape}\n"
            f"      shannon_entropy: shape {self.spectral.spectral_parameters.shannon_entropy.shape}\n"
            f"      tsallis_entropy: shape {self.spectral.spectral_parameters.tsallis_entropy.shape}\n"
            f"      spectral_crest_factor: shape {self.spectral.spectral_parameters.spectral_crest_factor.shape}\n"
            f"      spectral_centroid: shape {self.spectral.spectral_parameters.spectral_centroid.shape}\n"
            f"    )\n"
            f"  )\n"
            f")"
        )

def create_dummy_data():
    # Constants for dummy data
    n_samples = 1000  # 4 seconds at 250 Hz
    n_channels = 68   # Typical EEG montage
    n_freq_bins = 129 # Typical Welch PSD output
    bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    n_segments = 47   # As specified in example
    
    # Create dummy subjects
    subjects = {}
    
    for i in range(5):
        subject_id = f'subject_{i+1:03d}'
        
        # Create random raw segments
        raw_segments = np.random.randn(n_segments, n_samples, n_channels)
        
        # Create random PSD data
        psd = np.random.randn(n_segments, n_freq_bins)
        f = np.linspace(0, 125, n_freq_bins)  # Frequencies up to Nyquist (250/2 Hz)
        
        # Create random spectral parameters
        spectral_params = {
            'relative_powers': {band: np.random.randn(n_segments) for band in bands},
            'median_freq': np.random.randn(n_segments),
            'individual_alpha': np.random.randn(n_segments),
            'spectral_edge_frequency_95': np.random.randn(n_segments),
            'transition_frequency': np.random.randn(n_segments),
            'renyi_entropy': np.random.randn(n_segments),
            'shannon_entropy': np.random.randn(n_segments),
            'tsallis_entropy': np.random.randn(n_segments),
            'spectral_crest_factor': np.random.randn(n_segments),
            'spectral_centroid': np.random.randn(n_segments)
        }
        
        # Create subject using dataclass
        subjects[subject_id] = Subject(
            category='control' if i < 3 else 'patient',
            file_origin=f'{subject_id}.mat',
            sampling_rate=250,
            n_segments=n_segments,
            filtering={
                'type': 'bandpass',
                'band': [0.5, 70.0],
                'order': 4
            },
            ica_components_removed=np.random.randint(0, 5),
            raw_segments=raw_segments,
            spectral=Subject.SpectralData(
                psd=psd,
                f=f,
                spectral_parameters=Subject.SpectralData.SpectralParameters(
                    relative_powers=spectral_params['relative_powers'],
                    median_frequency=spectral_params['median_freq'],
                    individual_alpha_frequency=spectral_params['individual_alpha'],
                    spectral_edge_frequency_95=spectral_params['spectral_edge_frequency_95'],
                    transition_frequency=spectral_params['transition_frequency'],
                    renyi_entropy=spectral_params['renyi_entropy'],
                    shannon_entropy=spectral_params['shannon_entropy'],
                    tsallis_entropy=spectral_params['tsallis_entropy'],
                    spectral_crest_factor=spectral_params['spectral_crest_factor'],
                    spectral_centroid=spectral_params['spectral_centroid']
                )
            )
        )
    
    return subjects

# Create dummy data
dummy_subjects: dict[str, Subject] = create_dummy_data()

logger.info(f"dummy_subjects: {dummy_subjects['subject_001']}")

def create_dataset(path: str = 'h5test.h5'):
    with h5py.File(path, 'w') as f:

        f.attrs['description'] = 'Testing around'
        f.attrs['version'] = '1.0.0'
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['author'] = 'Alejandro'
        f.attrs['license'] = 'MIT'
        f.attrs['source_version'] = '1.0.0'
        
        # Add bands information with their frequency ranges
        f.attrs['relative_power_bands'] = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
        f.attrs['relative_power_frequencies'] = str({
            'delta': [0.5, 4.0],
            'theta': [4.0, 8.0],
            'alpha': [8.0, 13.0],
            'beta1': [13.0, 20.0],
            'beta2': [20.0, 30.0],
            'gamma': [30.0, 70.0]
        })
        f.attrs['spectral_parameters_interest_band'] = [0.5, 70.0]  # Same as filtering band

        subjects_group = f.create_group('subjects')

        for subject_id, subject in dummy_subjects.items():

            subj_group = subjects_group.create_group(subject_id)

            subj_group.attrs['category'] = subject.category
            subj_group.attrs['file_origin'] = subject.file_origin
            subj_group.attrs['sampling_rate'] = subject.sampling_rate
            subj_group.attrs['n_segments'] = subject.n_segments
            subj_group.attrs['filtering'] = json.dumps(subject.filtering)
            subj_group.attrs['ica_components_removed'] = subject.ica_components_removed

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

            relative_powers_group = spectral_params_group.create_group('relative_powers')
            for band, power in subject.spectral.spectral_parameters.relative_powers.items():
                relative_powers_group.create_dataset(band, data=power, compression='gzip')

            spectral_params_group.create_dataset('renyi_entropy', data=subject.spectral.spectral_parameters.renyi_entropy, compression='gzip')
            spectral_params_group.create_dataset('shannon_entropy', data=subject.spectral.spectral_parameters.shannon_entropy, compression='gzip')
            spectral_params_group.create_dataset('tsallis_entropy', data=subject.spectral.spectral_parameters.tsallis_entropy, compression='gzip')
            spectral_params_group.create_dataset('spectral_crest_factor', data=subject.spectral.spectral_parameters.spectral_crest_factor, compression='gzip')
            spectral_params_group.create_dataset('spectral_centroid', data=subject.spectral.spectral_parameters.spectral_centroid, compression='gzip')


if __name__ == '__main__':
    create_dataset()