from core.raw_dataset import RawDataset
from models.simple_2d import Simple2D3Layers
from loguru import logger
import wandb
from dotenv import load_dotenv
import os
from core.run_sweep import run_sweep
from typing import List, Tuple
import torch
import numpy as np

load_dotenv()

WANDB_PROJECT = "AD_vs_HC_sweep"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")

def _parse_two_level(s: str) -> list:
    s = s.strip()
    if "__" in s:
        return [[int(p) for p in g.split("_") if p] for g in s.split("__") if g]
    return [int(p) for p in s.split("_") if p]

def test_parser():
    print(_parse_two_level("16_32_64"))
    print(_parse_two_level("50_2__50_17__50_2"))
    print(_parse_two_level("1_2__1_17__1_1"))
    print(_parse_two_level("200_4__20_4__20_4"))
    print(_parse_two_level("200_8__20_8__20_2"))
    print(_parse_two_level("200_16__20_16__20_2"))
    print(_parse_two_level("200_32__20_32__20_2"))
    print(_parse_two_level("200_64__20_64__20_2"))
    print(_parse_two_level("200_128__20_128__20_2"))
    print(_parse_two_level("200_256__20_256__20_2"))

def main():
    with wandb.init(project=WANDB_PROJECT) as run:
        config = run.config

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        training_dataset = RawDataset(
            h5_file_path=H5_FILE_PATH,
            subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/training_subjects.txt",
            normalize=config.normalize
        )

        validation_dataset = RawDataset(
            h5_file_path=H5_FILE_PATH,
            subjects_txt_path="experiments/AD_vs_HC/combined/raw/splits/validation_subjects.txt",
            normalize=config.normalize
        )
        
        # Allow string-encoded params in sweeps (two-level parser)
        parsed_nf = _parse_two_level(config.n_filters)
        n_filters: List[int] = [int(x) for x in parsed_nf]

        parsed_ks = _parse_two_level(config.kernel_sizes)
        kernel_sizes: List[Tuple[int, int]] = [(int(p[0]), int(p[1])) for p in parsed_ks]

        parsed_strides = _parse_two_level(config.strides)
        strides: List[Tuple[int, int]] = [(int(p[0]), int(p[1])) for p in parsed_strides]

        model = Simple2D3Layers(
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dropout_rate=config.dropout_rate
        )

        run_sweep(model, run, training_dataset, validation_dataset)
       
if __name__ == "__main__":
    main()
    # test_parser()