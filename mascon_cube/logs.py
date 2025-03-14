from dataclasses import dataclass


@dataclass
class LogConfig:
    """Dataclass for logging"""

    log_every_n_epochs: int = 1
    draw_every_n_epochs: int = 50
