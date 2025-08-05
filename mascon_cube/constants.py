from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truths"
MESH_DIR = DATA_DIR / "3dmeshes"
VAL_DATASETS_DIR = DATA_DIR / "val_datasets"
TEST_DATASETS_DIR = DATA_DIR / "test_datasets"
OUTPUT_DIR = DATA_DIR / "output"
TENSORBOARD_DIR = ROOT_DIR / "tensorboard_logs"


# -------------------------------------------------------------------------------------------------------------------- #
# --------------- Constants used for plots in the paper to have nice and consistent colors and scales. --------------- #
# -------------------------------------------------------------------------------------------------------------------- #

__PLANETESIMAL_MASS_VMAX = 8.399126490844951e-06
__ITOKAWA_MASS_VMAX = 2.9296980406487892e-05
__BENNU_MASS_VMAX = 1.0714936263987457e-05
__EROS_MASS_VMAX = 4.031282754172379e-05
MASS_VMAX: dict[str, float] = {
    "planetesimal": __PLANETESIMAL_MASS_VMAX,
    "planetesimal_decentered": __PLANETESIMAL_MASS_VMAX,
    "planetesimal_uniform": __PLANETESIMAL_MASS_VMAX,
    "itokawa_cos": __ITOKAWA_MASS_VMAX,
    "itokawa": __ITOKAWA_MASS_VMAX,
    "bennu": __BENNU_MASS_VMAX,
    "eros_uniform": __EROS_MASS_VMAX,
    "eros_2": __EROS_MASS_VMAX,
    "eros_3": __EROS_MASS_VMAX,
}

__PLANETESIMAL_DENSITY_VMAX = 1.02
__ITOKAWA_DENSITY_VMAX = 3.55
__BENNU_DENSITY_VMAX = 1.30
__EROS_DENSITY_VMAX = 4.89
DENSITY_VMAX: dict[str, float] = {
    "planetesimal": __PLANETESIMAL_DENSITY_VMAX,
    "planetesimal_decentered": __PLANETESIMAL_DENSITY_VMAX,
    "planetesimal_uniform": __PLANETESIMAL_DENSITY_VMAX,
    "itokawa_cos": __ITOKAWA_DENSITY_VMAX,
    "itokawa": __ITOKAWA_DENSITY_VMAX,
    "bennu": __BENNU_DENSITY_VMAX,
    "eros_uniform": __EROS_DENSITY_VMAX,
    "eros_2": __EROS_DENSITY_VMAX,
    "eros_3": __EROS_DENSITY_VMAX,
}
