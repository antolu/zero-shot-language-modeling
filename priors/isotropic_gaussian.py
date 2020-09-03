import logging
from typing import Union

import torch

from .prior import Prior

log = logging.getLogger(__name__)


class IsotropicGaussianPrior(Prior):
    """
    Isotropic Gaussian Prior. Yields a mean regularisation for the model during training.
    This prior corresponds to N(w*, I).
    """
    def __init__(self, model: torch.nn.Module):
        """
        Parameters
        ----------
        model : nn.Module
            The model to optimized parameters from, to use for the penalty term calculation.
        """
        super(IsotropicGaussianPrior, self).__init__()

        self._means = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.clone().detach()

    def penalty(self, model, *args) -> Union[torch.Tensor, int]:
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
