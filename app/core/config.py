import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Anti-spoofing model and config paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2_spoof_model.h5")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "model_config.json")

