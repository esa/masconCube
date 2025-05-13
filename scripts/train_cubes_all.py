import os
import queue
import shutil
import subprocess
import threading
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from mascon_cube.constants import GROUND_TRUTH_DIR


def main(gpus: list[int]):
    asteroids = [dir.name for dir in Path(GROUND_TRUTH_DIR).iterdir() if dir.is_dir()]

    assert len(gpus) >= 1, "No GPUs available"

    asteroid_queue = queue.Queue()
    for asteroid in asteroids:
        asteroid_queue.put(asteroid)

    progress_bar = tqdm(
        total=len(asteroids), desc="Trainings completed", position=0, leave=True
    )
    progress_lock = threading.Lock()  # To protect tqdm updates
    current_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    if current_env is None:
        raise RuntimeError("This script must be run in a conda environment.")
    conda_path = shutil.which("conda")
    if conda_path is None:
        raise FileNotFoundError(
            "Cannot find 'conda' in PATH. Ensure you're running inside a conda environment."
        )

    def gpu_worker(gpu_id: int):
        while not asteroid_queue.empty():
            try:
                asteroid = asteroid_queue.get_nowait()
            except queue.Empty:
                return

            tqdm.write(f"[GPU {gpu_id}] Starting: {asteroid}")
            env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
            train_script = Path(__file__).parent / "train.py"
            process = subprocess.Popen(
                [
                    conda_path,
                    "run",
                    "-n",
                    current_env,
                    "python",
                    str(train_script),
                    asteroid,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
            )
            process.wait()
            tqdm.write(f"[GPU {gpu_id}] Finished: {asteroid}")
            with progress_lock:
                progress_bar.update(1)

            asteroid_queue.task_done()

    # Create one thread per GPU
    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=gpu_worker, args=(gpu_id,))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    progress_bar.close()
    print("All trainings complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train mascon cubes on multiple GPUs")
    parser.description = (
        "This script trains mascon cubes on multiple GPUs. "
        "It will automatically distribute the training across the available GPUs."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU IDs to use for training (e.g., 0 1 2)",
    )
    args = parser.parse_args()
    main(args.gpus)
