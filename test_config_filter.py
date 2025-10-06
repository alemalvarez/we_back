#!/usr/bin/env python3

import sys
from typing import Dict, Any, Type, TypeVar

from core.schemas import OptimizerConfig, CriterionConfig

T = TypeVar('T')

def filter_config_params(config_class: Type[T], config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter config dict to only include parameters expected by the config class.

    This allows WarnUnsetDefaultsModel to warn about missing parameters that fall back to defaults.
    """
    expected_params = set(config_class.__annotations__.keys())
    return {k: v for k, v in config_dict.items() if k in expected_params}

def main():
    # Test config dict like from YAML
    config_dict = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'pos_weight': 0.7,
        'unknown_param': 'should_be_ignored'
    }

    print("Testing filter_config_params function...")

    # Test filtering for OptimizerConfig
    optimizer_params = filter_config_params(OptimizerConfig, config_dict)
    print('Optimizer params:', optimizer_params)

    # Test that warnings are shown for OptimizerConfig
    print("\nCreating OptimizerConfig with filtered params...")
    optimizer_config = OptimizerConfig(**optimizer_params)
    print('Optimizer config created successfully')

    # Test filtering for CriterionConfig
    criterion_params = filter_config_params(CriterionConfig, config_dict)
    print('\nCriterion params:', criterion_params)

    # Test that warnings are shown for CriterionConfig
    print("\nCreating CriterionConfig with filtered params...")
    criterion_config = CriterionConfig(**criterion_params)
    print('Criterion config created successfully')

if __name__ == "__main__":
    main()
