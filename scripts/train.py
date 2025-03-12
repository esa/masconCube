import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mascon_cube.constants import OUTPUT_DIR
from mascon_cube.data.mascon_model import get_mascon_model
from mascon_cube.data.sampling import get_target_point_sampler
from mascon_cube.losses import normalized_L1_loss
from mascon_cube.models import MasconCube
from mascon_cube.training import TrainingConfig, training_loop


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asteroid = "itokawa_lp"

    cube = MasconCube(100, asteroid, device=device)
    gt = get_mascon_model(asteroid, device=device)
    data_sampler = get_target_point_sampler(
        n=1000,
        asteroid_mesh=asteroid,
        method="spherical",
        bounds=(0.5, 1.5),
        device=device,
    )
    optimizer = Adam([cube.weights], lr=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=200, min_lr=1e-8)

    # Training configuration
    config = TrainingConfig(
        n_epochs=500,
        n_epochs_before_resampling=10,
        loss_fn=normalized_L1_loss,
        data_sampler=data_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        use_tensorboard=True,
    )

    # Train the cube
    trained_cube = training_loop(cube, gt, config)

    # Save the trained cube
    torch.save(trained_cube, OUTPUT_DIR / "trained_cube.pt")


if __name__ == "__main__":
    main()
