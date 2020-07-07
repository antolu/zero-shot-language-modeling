import logging
import time
from copy import deepcopy
from typing import Union

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import SplitCrossEntropyLoss
from regularisation import LockedDropout, WeightDrop

log = logging.getLogger(__name__)


class LaplacePrior:
    def __init__(self, model: torch.nn.Module, loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
                 dataloader: DataLoader, use_apex: bool = False, amp=None, optimizer: torch.optim.Optimizer = None,
                 device: Union[torch.device, str] = 'cpu'):

        self.model = model
        self.loss_function = loss_function
        self.dataloader = dataloader
        self.use_apex = use_apex
        self.amp = amp
        self.optimizer = optimizer
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.clone().detach()

        # need to be in train mode to calculate gradients
        self.model.train()

        # deactivate dropout to calculate full gradients
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0
                print(m)
            elif isinstance(m, LockedDropout) or isinstance(m, WeightDrop):
                m.dropout = 0
                print(m)

        batch = 0
        log.info("Starting calculation of Fisher's matrix")
        start_time = time.time()

        with tqdm(self.dataloader, dynamic_ncols=True) as pbar:
            for inputs, targets, seq_len, lang in pbar:
                inputs.squeeze(0).to(self.device)
                targets.squeeze(0).to(self.device)

                hidden = self.model.init_hidden(inputs.size(-1))
                self.model.zero_grad()
                output, hidden = self.model(inputs, hidden, lang)

                loss = self.loss_function(self.model.decoder.weight, self.model.decoder.bias, output, targets)
                if self.use_apex and self.optimizer:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                batch += 1

                pbar.update(1)

                for n, p in self.params.items():
                    if p.grad is not None:
                        precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: p / batch for n, p in precision_matrices.items()}
        log.info("Finished calculation of Fisher's matrix, it took {} minutes".format(
            (time.time() - start_time) / 1000 / 60))
        return precision_matrices

    def penalty(self, model, *args) -> Union[torch.Tensor, int]:
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
