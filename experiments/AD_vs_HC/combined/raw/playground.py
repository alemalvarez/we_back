import os
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
from core.model_playground import run_model_playground


load_dotenv()

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "h5test_raw_only.h5")
logger.info(f"H5 file path: {H5_FILE_PATH}")


configs_to_test = sorted(
    os.path.join("configs", f) for f in os.listdir(os.path.join(os.path.dirname(__file__), "configs")) if f.endswith(".yaml")
)

if __name__ == "__main__":
    run_model_playground(
        config_filenames=configs_to_test,
        n_folds=5,
        output_dir="playground_results",
        h5_file_path=H5_FILE_PATH
    )
