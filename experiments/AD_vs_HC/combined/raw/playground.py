import os
from dotenv import load_dotenv
from loguru import logger

from core.model_playground import run_model_playground

load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")
configs_to_test = [
    "configs/fire_lammini_style.yaml",
    "configs/nice_for_layer.yaml",
    "configs/supergood.yaml"
]

if __name__ == "__main__":
    run_model_playground(
        config_filenames=configs_to_test,
        n_folds=5,
        output_dir="playground_results",
        h5_file_path=H5_FILE_PATH
    )
