from pathlib import Path
from src.utils.helpers import load_config as _load_config


def load_config_from_root(filename: str = "config.yaml") -> dict:
    path = Path(__file__).parent / filename
    return _load_config(path)
