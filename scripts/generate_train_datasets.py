import os
from argparse import ArgumentParser

from mascon_cube.constants import GROUND_TRUTH_DIR
from mascon_cube.data.datasets import create_random_dataset


def get_train_datasets():
    pass


if __name__ == "__main__":
    asteroids = [f.name for f in os.scandir(GROUND_TRUTH_DIR) if f.is_dir()]
    parser = ArgumentParser()
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for dataset generation"
    )
    parser.add_argument(
        "--n-points",
        "-n",
        type=int,
        default=10000,
        help="Number of points in the training dataset",
    )
    parser.add_argument(
        "--sampling-method",
        "-sm",
        type=str,
        default="spherical",
        help="Sampling method to use for dataset generation",
    )
    parser.add_argument(
        "--range", "-r", type=float, nargs=2, default=(0.0, 2.0), help="Sampling range"
    )
    args = parser.parse_args()
    for asteroid in asteroids:
        dataset_path = create_random_dataset(
            asteroid=asteroid,
            n=args.n_points,
            sampling_method=args.sampling_method,
            sampling_min=args.range[0],
            sampling_max=args.range[1],
            seed=args.seed,
        )
        print(f"Created training dataset for {asteroid} at {dataset_path}")
