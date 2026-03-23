import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config, config_path="config.yaml"):
    path = Path(config_path)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
