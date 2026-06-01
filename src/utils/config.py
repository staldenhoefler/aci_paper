"""Load params.yaml and expose the run ID for consistent output naming."""
import os
from datetime import datetime
from pathlib import Path

import yaml

_PARAMS_PATH = Path(__file__).parent.parent.parent / "params.yaml"


def load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def params_path() -> Path:
    return _PARAMS_PATH


def run_id() -> str:
    """Return the shared run ID (set by run_all.py or a fresh timestamp)."""
    return os.environ.get("ACI_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
