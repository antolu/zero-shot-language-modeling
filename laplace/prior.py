import torch


class _Prior:
    """
    Abstract class to represent any prior to be used with MAP training.
    """
    def penalty(self, *args) -> torch.Tensor:
        raise NotImplementedError
