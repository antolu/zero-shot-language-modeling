from typing import Union

import torch

from laplace.prior import _Prior


class GaussianPrior(_Prior):
    """
    A prior with a penalty corresponding to a Gaussian.
    """
    # TODO: variable variance
    @classmethod
    def penalty(cls, model: torch.nn.Module, *args) -> Union[torch.Tensor, int]:
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = p ** 2
                loss += _loss.sum()
        return loss
