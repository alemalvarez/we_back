
from dataclasses import dataclass
from typing_extensions import Type, TypeVar
import numpy as np
from typing import Optional, Tuple, Dict, Any
from loguru import logger
from typing import Literal
from pydantic import BaseModel, model_validator

class _UnsetSentinel:
    pass

_UNSET = _UnsetSentinel()

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
    source_folder: str
    folder_id: str
    is_eeg: bool

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
            f"  source_folder: {self.source_folder}\n"
            f"  folder_id: {self.folder_id}\n"
            f"  is_eeg: {self.is_eeg}\n"
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

class WarnUnsetDefaultsModel(BaseModel):
    _warnings_shown: bool = False

    @model_validator(mode="after")
    def _warn_unset_defaults(self):
        # Only show warnings once per instance to avoid duplicate warnings
        # when the model is used as a nested field in another model
        if self._warnings_shown:
            return self

        model_fields = type(self).model_fields
        for name in model_fields:
            if name not in self.model_fields_set:
                logger.warning(
                    f"{self.__class__.__name__}.{name} left unset; using default: {getattr(self, name)!r}"
                )

        # Mark that warnings have been shown for this instance
        object.__setattr__(self, '_warnings_shown', True)
        return self


class BaseModelConfig(WarnUnsetDefaultsModel):
    random_seed: int
    model_name: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    early_stopping_metric: Optional[Literal['loss', 'f1', 'mcc', 'kappa']] = 'loss' # loss is generaly recommended.
    use_cosine_annealing: bool = False

    # defaults stay as field defaults; validator warns only when omitted

class NetworkConfig(WarnUnsetDefaultsModel):
    model_name: str

class OptimizerConfig(WarnUnsetDefaultsModel):
    learning_rate: float
    weight_decay: Optional[float] = None
    use_cosine_annealing: bool = False
    cosine_annealing_t_0: int = 5
    cosine_annealing_t_mult: int = 1
    cosine_annealing_eta_min: float = 1e-6


class CriterionConfig(WarnUnsetDefaultsModel):
    pos_weight_type: Literal['fixed', 'multiplied'] = 'fixed'
    pos_weight_value: float = 1.0

class DatasetConfig(WarnUnsetDefaultsModel):
    h5_file_path: str
    dataset_names: list[str]

class SpectralDatasetConfig(DatasetConfig):
    dataset_type: Literal['spectral'] = 'spectral'
    spectral_normalization: Literal['min-max', 'standard', 'none'] = 'none'

class RawDatasetConfig(DatasetConfig):
    dataset_type: Literal['raw'] = 'raw'
    raw_normalization: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full'] = 'sample-channel'
    augment: bool = False
    augment_prob: Tuple[float, float] = (0.5, 0.0)
    noise_std: float = 0.1

class MultiDatasetConfig(DatasetConfig):
    dataset_type: Literal['multi'] = 'multi'
    raw_normalization: Literal['sample-channel', 'sample', 'channel-subject', 'subject', 'channel', 'full'] = 'sample-channel'
    spectral_normalization: Literal['min-max', 'standard', 'none'] = 'none'


class RunConfig(WarnUnsetDefaultsModel):
    network_config: NetworkConfig
    optimizer_config: OptimizerConfig
    criterion_config: CriterionConfig
    dataset_config: DatasetConfig
    random_seed: int
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    early_stopping_metric: Optional[Literal['loss', 'f1', 'mcc', 'kappa']] = 'loss' # loss is generaly recommended.
    log_to_wandb: bool = False
    wandb_init: Optional[dict] = None

    @model_validator(mode="after")
    def _print_config_summary(self):
        logger.info("=" * 80)
        logger.info("RunConfig Summary")
        logger.info("=" * 80)
        
        model_fields = type(self).model_fields
        for name in model_fields:
            value = getattr(self, name)
            is_overridden = name in self.model_fields_set
            status = "OVERRIDDEN" if is_overridden else "DEFAULT"
            
            # Format nested configs nicely
            if isinstance(value, WarnUnsetDefaultsModel):
                logger.info(f"{name}: [{status}]")
                self._print_nested_config(value, indent="  ")
            else:
                logger.info(f"{name}: {value!r} [{status}]")
        
        logger.info("=" * 80)
        return self
    
    def _print_nested_config(self, config: WarnUnsetDefaultsModel, indent: str = ""):
        model_fields = type(config).model_fields
        for name in model_fields:
            value = getattr(config, name)
            is_overridden = name in config.model_fields_set
            status = "OVERRIDDEN" if is_overridden else "DEFAULT"
            logger.info(f"{indent}{name}: {value!r} [{status}]")

T = TypeVar('T')

def filter_config_params(config_class: Type[T], config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter config dict to only include parameters expected by the config class.

    This allows WarnUnsetDefaultsModel to warn about missing parameters that fall back to defaults.
    """
    expected_params = set(config_class.__annotations__.keys())
    return {k: v for k, v in config_dict.items() if k in expected_params}
