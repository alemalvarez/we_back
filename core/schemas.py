from dataclasses import dataclass
import numpy as np

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
            relative_powers: np.ndarray
            renyi_entropy: np.ndarray
            shannon_entropy: np.ndarray
            tsallis_entropy: np.ndarray
            spectral_crest_factor: np.ndarray
            spectral_centroid: np.ndarray
            spectral_bandwidth: np.ndarray
            
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
    trial_length_secs: float

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
            f"      relative_powers: shape {self.spectral.spectral_parameters.relative_powers.shape}\n"
            f"      renyi_entropy: shape {self.spectral.spectral_parameters.renyi_entropy.shape}\n"
            f"      shannon_entropy: shape {self.spectral.spectral_parameters.shannon_entropy.shape}\n"
            f"      tsallis_entropy: shape {self.spectral.spectral_parameters.tsallis_entropy.shape}\n"
            f"      spectral_crest_factor: shape {self.spectral.spectral_parameters.spectral_crest_factor.shape}\n"
            f"      spectral_centroid: shape {self.spectral.spectral_parameters.spectral_centroid.shape}\n"
            f"      spectral_bandwidth: shape {self.spectral.spectral_parameters.spectral_bandwidth.shape}\n"
            f"    )\n"
            f"  )\n"
            f")"
        )