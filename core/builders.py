from torch import nn
import torch
import torch.optim as optim
from core.schemas import NetworkConfig, OptimizerConfig, CriterionConfig
from models.simple_2d import Simple2D3Layers, DeeperCustom, Deeper2D, Improved2D
from models.spectral_net import SpectralNet

def build_model(config: NetworkConfig) -> nn.Module: 
    name_class_map = {
        "DeeperCustom": DeeperCustom,
        "Deeper2D": Deeper2D,
        "Improved2D": Improved2D,
        "Simple2D3Layers": Simple2D3Layers,
        "SpectralNet": SpectralNet,
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