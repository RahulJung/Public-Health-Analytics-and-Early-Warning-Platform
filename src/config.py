from pathlib import Path
import yaml

# Resolve shared project paths once so all modules read the same config file.
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config():
    # Load YAML configuration for data paths, API settings, and model parameters.
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
