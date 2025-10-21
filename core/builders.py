from typing import List, Optional
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from core.multi_dataset import MultiDataset
from core.schemas import NetworkConfig, OptimizerConfig, CriterionConfig, DatasetConfig, SpectralDatasetConfig, RawDatasetConfig, MultiDatasetConfig
from models.concatter import Concatter, GatedConcatter, SimpleConcatter
from models.simple_2d import Simple2D3Layers, DeeperCustom, Deeper2D, Improved2D
from models.spectral_net import SpectralNet
from models.spectral_net import AdvancedSpectralNet
from core.spectral_dataset import SpectralDataset
from core.raw_dataset import RawDataset
from models.squeezer import DeeperSE

def build_model(config: NetworkConfig) -> nn.Module: 
    name_class_map = {
        "DeeperCustom": DeeperCustom,
        "Deeper2D": Deeper2D,
        "Improved2D": Improved2D,
        "Simple2D3Layers": Simple2D3Layers,
        "SpectralNet": SpectralNet,
        "AdvancedSpectralNet": AdvancedSpectralNet,
        "Concatter": Concatter,
        "GatedConcatter": GatedConcatter,
        "SimpleConcatter": SimpleConcatter,
        "DeeperSE": DeeperSE,
    }
    return name_class_map[config.model_name](config)

def build_optimizer(config: OptimizerConfig, model: nn.Module) -> optim.Optimizer:
    if config.weight_decay is not None:
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        return optim.Adam(model.parameters(), lr=config.learning_rate)

def build_scheduler(config: OptimizerConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None:
    if config.use_cosine_annealing:
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.cosine_annealing_t_0, T_mult=config.cosine_annealing_t_mult, eta_min=config.cosine_annealing_eta_min)
    else:
        return None

def build_criterion(config: CriterionConfig, dataset_pos_neg: tuple[int, int]) -> nn.Module:
    if config.pos_weight_type == 'fixed':
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight_value))
    else:
        pos_weight = config.pos_weight_value * (dataset_pos_neg[0] / dataset_pos_neg[1])
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

def build_dataset(config: DatasetConfig, subjects_path: Optional[str] = None, subjects_list: Optional[List[str]] = None, validation: bool = False) -> Dataset:
    if isinstance(config, SpectralDatasetConfig):
        return SpectralDataset(
            config.h5_file_path,
            subjects_path,
            config.spectral_normalization,
            subjects_list
        )
    elif isinstance(config, MultiDatasetConfig):
        return MultiDataset(
            config.h5_file_path,
            subjects_path,
            config.raw_normalization,
            config.spectral_normalization,
            subjects_list
        )
    else:
        assert isinstance(config, RawDatasetConfig)
        if validation:
            return RawDataset(
                h5_file_path=config.h5_file_path,
                subjects_txt_path=subjects_path,
                normalize=config.raw_normalization,
                augment=False,
                augment_prob=config.augment_prob,
                noise_std=config.noise_std,
                subjects_list=subjects_list
            )
        else:
            return RawDataset(
                h5_file_path=config.h5_file_path,
                subjects_txt_path=subjects_path,
                normalize=config.raw_normalization,
                augment=config.augment,
                augment_prob=config.augment_prob,
                noise_std=config.noise_std,
                subjects_list=subjects_list
            )
