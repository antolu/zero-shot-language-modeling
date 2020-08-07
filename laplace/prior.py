import torch
from torch.utils.tensorboard import SummaryWriter


class Prior:
    """
    Abstract class to represent any prior to be used with MAP training.
    """
    def penalty(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def write_nts(self, tbwriter: SummaryWriter):
        raise NotImplementedError
