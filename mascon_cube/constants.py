from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truths"
MESH_DIR = DATA_DIR / "3dmeshes"
TENSORBOARD_DIR = ROOT_DIR / "tensorboard_logs"
