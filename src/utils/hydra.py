import os
from pathlib import Path

from omegaconf import DictConfig


def extract_output_dir(config: DictConfig) -> Path:
    """
    Extracts path to output directory as pathlib.Path and ensures it exists.
    """
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))

    output_dir = Path(os.environ['OUTPUT_DIR']) / date

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir



def preprocess_config(config: DictConfig):
    """
    Sets config.exp.log_dir to date-extracted output path.
    """
    config.exp.log_dir = extract_output_dir(config)
