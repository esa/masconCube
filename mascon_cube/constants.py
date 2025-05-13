from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truths"
MESH_DIR = DATA_DIR / "3dmeshes"
VAL_DATASETS_DIR = DATA_DIR / "val_datasets"
TEST_DATASETS_DIR = DATA_DIR / "test_datasets"
OUTPUT_DIR = DATA_DIR / "output"
TENSORBOARD_DIR = ROOT_DIR / "tensorboard_logs"
