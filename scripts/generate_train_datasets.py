import os
import sys
from argparse import ArgumentParser

from mascon_cube.constants import GROUND_TRUTH_DIR
from mascon_cube.data.datasets import create_random_dataset, create_trajectory_dataset


def random():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for dataset generation"
    )
    parser.add_argument(
        "--n-points",
        "-n",
        type=int,
        default=100_000,
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
        "--range", "-r", type=float, nargs=2, default=(0.0, 1.0), help="Sampling range"
    )
    args = parser.parse_args(sys.argv[2:])
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


def traj(asteroids):
    parser = ArgumentParser()
    parser.add_argument(
        "--n-points",
        "-n",
        type=int,
        default=100_000,
        help="Number of points in the training dataset",
    )
    parser.add_argument(
        "--safety-coefficient",
        "-sc",
        type=float,
        default=1.4,
        help="Safety coefficient for trajectory simulation",
    )
    parser.add_argument(
        "--exit-radius",
        "-er",
        type=float,
        default=2.0,
        help="Exit radius for trajectory simulation",
    )
    parser.add_argument(
        "--propagation-days",
        "-pd",
        type=float,
        default=1.0,
        help="Propagation days for trajectory simulation",
    )
    parser.add_argument(
        "--orb-params",
        "-op",
        type=float,
        nargs=6,
        default=(1.5, 0.0, 3.1415 / 2, 0.0, 0.0, 3.1415 / 2),
        help="Starting orbital parameters for trajectory-based dataset generation",
    )
    args = parser.parse_args(sys.argv[2:])
    for asteroid in asteroids:
        dataset_path = create_trajectory_dataset(
            asteroid=asteroid,
            n=args.n_points,
            safety_coefficient=args.safety_coefficient,
            exit_radius=args.exit_radius,
            starting_orb_params=(args.orb_params,),
            propagation_days=args.propagation_days,
        )
        print(f"Created training dataset for {asteroid} at {dataset_path}")


if __name__ == "__main__":
    asteroids = [f.name for f in os.scandir(GROUND_TRUTH_DIR) if f.is_dir()]
    parser = ArgumentParser()
    parser.add_argument(
        "method",
        type=str,
        choices=["random", "traj"],
        help="Dataset generation method",
    )
    args = parser.parse_args(sys.argv[1:2])
    if args.method == "random":
        random(asteroids)
    elif args.method == "traj":
        traj(asteroids)
    else:
        print("Unknown method.")
        parser.print_help()
        sys.exit(1)
