from dataclasses import dataclass

import lazy_import
import torch

tensorboard = lazy_import.lazy_module("torch.utils.tensorboard")


@dataclass
class LogConfig:
    """Dataclass for logging"""

    log_every_n_epochs: int = 1
    draw_every_n_epochs: int = 50


class SummaryWriter(tensorboard.SummaryWriter):
    """Override the SummaryWriter class to fix a bug in the add_hparams method."""

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = tensorboard.summary.hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)
