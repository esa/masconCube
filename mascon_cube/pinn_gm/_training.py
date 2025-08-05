from mascon_cube.training import AbstractTrainingConfig


class PinnTrainingConfig(AbstractTrainingConfig):
    """Dataclass for training configuration for PINN models"""

    # from superclass:
    n_epochs: int = 8192  ## override
    loss_fn: str = "normalized_l1_loss"  #  'loss_fcns': [['percent', 'rms']],
    batch_size: int = 2048
    sampling_method: str = "spherical"
    sampling_min: float = 0.0
    sampling_max: float = 2.4
    lr: float = 0.00390625
    scheduler_factor: float = 1.0
    scheduler_patience: int = 8192
    scheduler_min_lr: float = 0.00390625

    # New
