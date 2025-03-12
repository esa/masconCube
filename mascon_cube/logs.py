from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LogConfig:
    """Dataclass for logging"""

    log_every_n_epochs: int = 1
    val_every_n_epochs: int = 50
    draw_every_n_epochs: int = 50
    val_dataset: Optional[torch.Tensor] = None
