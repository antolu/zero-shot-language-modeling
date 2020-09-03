import logging
import math
import time
from copy import deepcopy
from typing import Union
from os import path

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from criterion import SplitCrossEntropyLoss
from priors import Prior
from regularisation import LockedDropout, WeightDrop

log = logging.getLogger(__name__)


def softrelu(x: torch.Tensor):
    return torch.log1p(torch.exp(x))


def kl_div(mu: torch.Tensor, log_var: torch.Tensor):
    prior_sigma = 1
    kl = math.log(prior_sigma) - log_var.sum() - mu.numel() + ((log_var.exp() + mu ** 2).sum() / prior_sigma)
    kl = 0.5 * kl
    return kl


class VIPrior(Prior):
    def __init__(self, model: torch.nn.Module, device: Union[torch.device, str] = 'cpu', **kwargs):
        super(VIPrior, self).__init__()

        self.model = model
        self.device = device

        self.noise = torch.distributions.normal.Normal(loc=torch.tensor(0.).to(device), scale=torch.tensor(1.).to(device))

        self.params = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self._means = {n: nn.Parameter(torch.zeros_like(p)) for n, p in model.named_parameters() if p.requires_grad}
        self._log_variance = {n: nn.Parameter(torch.ones_like(p)) for n, p in model.named_parameters() if p.requires_grad}

        for _, p in self._means.items():
            nn.init.uniform_(p, -1, 1)
        for _, p in self._log_variance.items():
            nn.init.uniform_(p, -5, -3)

    def kl_div(self) -> torch.Tensor:
        kl = 0

        for n in self.params:
            kl += kl_div(self._means[n], self._log_variance[n])

        return kl

    def sample_weights(self, model: nn.Module):
        for n in self.params:
            sampled_weight = self._means[n] + softrelu(self._log_variance[n]) * self.noise.sample(self._log_variance[n].shape)
            model._parameters[n] = sampled_weight

    def penalty(self, model, *args) -> Union[torch.Tensor, int]:
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = (p - self._means[n])**2 * self._log_variance[n].exp()
                loss += _loss.sum()
        return loss

    def calculate_nts(self):
        output = dict()
        for n in self.params:
            output[n] = (torch.abs(self._means[n]) / (self._log_variance[n] / 2)).exp().clone().detach().cpu().flatten()

        return output