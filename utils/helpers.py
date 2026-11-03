import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve(config_path) -> Path:
    path = Path(config_path)
    return path if path.is_absolute() or path.exists() else ROOT / path


def load_config(config_path="config.yaml"):
    path = _resolve(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config, config_path="config.yaml"):
    with open(_resolve(config_path), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
