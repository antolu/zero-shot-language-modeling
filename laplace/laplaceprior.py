import logging
import time
from copy import deepcopy
from typing import Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import SplitCrossEntropyLoss
from .prior import Prior
from regularisation import LockedDropout, WeightDrop

log = logging.getLogger(__name__)


class LaplacePrior(Prior):
    def __init__(self, model: torch.nn.Module, loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
                 dataloader: DataLoader, use_apex: bool = False, amp=None, optimizer: torch.optim.Optimizer = None,
                 device: Union[torch.device, str] = 'cpu'):
        super(LaplacePrior, self).__init__()

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = _diag_fisher(model, loss_function, dataloader, optimizer, amp, device)

        for n, p in params.items():
            self._means[n] = p.clone().detach()

    def penalty(self, model, *args) -> Union[torch.Tensor, int]:
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss


def _diag_fisher(model: nn.Module, loss_function: torch.nn.modules.loss._Loss,
                 dataloader: DataLoader, optimizer: torch.optim.Optimizer, amp=None, device: Union[torch.device, str] = 'cpu'):
    precision_matrices = {}

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in params.items():
        p.data.zero_()
        precision_matrices[n] = p.clone().detach()

    # need to be in train mode to calculate gradients
    model.train()

    # deactivate dropout to calculate full gradients
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
            print(m)
        elif isinstance(m, LockedDropout) or isinstance(m, WeightDrop):
            m.dropout = 0
            print(m)

    batch = 0
    log.info("Starting calculation of Fisher's matrix")
    start_time = time.time()

    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        for inputs, targets, seq_len, lang in pbar:
            inputs = inputs.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)

            hidden = model.init_hidden(inputs.size(-1))
            model.zero_grad()
            output, hidden = model(inputs, hidden, lang)

            loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
            if amp is not None and optimizer is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            batch += 1

            pbar.update(1)

            for n, p in params.items():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2

    precision_matrices = dict([(n, p / batch) for n, p in precision_matrices.items()])
    log.info("Finished calculation of Fisher's matrix, it took {} minutes".format(
        (time.time() - start_time) / (1000 / 60)))
    return precision_matrices
