from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass 
class Subject:
    @dataclass
    class SpectralData:

        @dataclass
        class SpectralParameters:
            median_frequency: Optional[np.ndarray]
            spectral_edge_frequency_95: Optional[np.ndarray]
            individual_alpha_frequency: Optional[np.ndarray]
            transition_frequency: Optional[np.ndarray]
            relative_powers: Optional[np.ndarray]
            renyi_entropy: Optional[np.ndarray]
            shannon_entropy: Optional[np.ndarray]
            tsallis_entropy: Optional[np.ndarray]
            spectral_crest_factor: Optional[np.ndarray]
            spectral_centroid: Optional[np.ndarray]
            spectral_bandwidth: Optional[np.ndarray]

        psd: Optional[np.ndarray]
        f: Optional[np.ndarray]
        spectral_parameters: Optional[SpectralParameters]

    # Identifying information
    category: str
    file_origin: str

    # Data information
    sampling_rate: int
    n_segments: int
    filtering: dict
    ica_components_removed: int
    trial_length_secs: float
    points_per_segment: int
    
    raw_segments: Optional[np.ndarray]
    spectral: Optional[SpectralData]

    def __str__(self) -> str:
        """String representation of the Subject class."""
        result = (
            f"Subject(\n"
            f"  category: {self.category}\n"
            f"  file_origin: {self.file_origin}\n"
            f"  sampling_rate: {self.sampling_rate} Hz\n"
            f"  n_segments: {self.n_segments}\n"
            f"  filtering: {self.filtering}\n"
            f"  trial_length_secs: {self.trial_length_secs}\n"
            f"  ica_components_removed: {self.ica_components_removed}\n"
            f"  points_per_segment: {self.points_per_segment}\n"
        )

        if self.raw_segments is not None:
            result += f"  raw_segments: shape {self.raw_segments.shape}\n"
        else:
            result += "  raw_segments: None\n"

        if self.spectral is not None:
            result += "  spectral: SpectralData(\n"
            if self.spectral.psd is not None:
                result += f"    psd: shape {self.spectral.psd.shape}\n"
            else:
                result += "    psd: None\n"

            if self.spectral.f is not None:
                result += f"    f: shape {self.spectral.f.shape}\n"
            else:
                result += "    f: None\n"

            if self.spectral.spectral_parameters is not None:
                result += "    spectral_parameters: SpectralParameters(\n"
                params = self.spectral.spectral_parameters
                if params.median_frequency is not None:
                    result += f"      median_frequency: shape {params.median_frequency.shape}\n"
                if params.spectral_edge_frequency_95 is not None:
                    result += f"      spectral_edge_frequency_95: shape {params.spectral_edge_frequency_95.shape}\n"
                if params.individual_alpha_frequency is not None:
                    result += f"      individual_alpha_frequency: shape {params.individual_alpha_frequency.shape}\n"
                if params.transition_frequency is not None:
                    result += f"      transition_frequency: shape {params.transition_frequency.shape}\n"
                if params.relative_powers is not None:
                    result += f"      relative_powers: shape {params.relative_powers.shape}\n"
                if params.renyi_entropy is not None:
                    result += f"      renyi_entropy: shape {params.renyi_entropy.shape}\n"
                if params.shannon_entropy is not None:
                    result += f"      shannon_entropy: shape {params.shannon_entropy.shape}\n"
                if params.tsallis_entropy is not None:
                    result += f"      tsallis_entropy: shape {params.tsallis_entropy.shape}\n"
                if params.spectral_crest_factor is not None:
                    result += f"      spectral_crest_factor: shape {params.spectral_crest_factor.shape}\n"
                if params.spectral_centroid is not None:
                    result += f"      spectral_centroid: shape {params.spectral_centroid.shape}\n"
                if params.spectral_bandwidth is not None:
                    result += f"      spectral_bandwidth: shape {params.spectral_bandwidth.shape}\n"
                result += "    )\n"
            else:
                result += "    spectral_parameters: None\n"
            result += "  )\n"
        else:
            result += "  spectral: None\n"

        result += ")"
        return result