import logging
from copy import deepcopy
from typing import Union

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from criterion import SplitCrossEntropyLoss
from data import DataLoader
from model import LSTM
from regularisation import LockedDropout, WeightDrop

log = logging.getLogger(__name__)


class BoringPrior:
    @classmethod
    def penalty(cls, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = p ** 2
                loss += _loss.sum()
        return loss


class EWC:
    def __init__(self, model: LSTM, loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
                 dataloader: DataLoader, use_apex: bool = False, amp=None):

        self.model = model
        self.loss_function = loss_function
        self.dataloader = dataloader
        self.use_apex = use_apex
        self.amp = amp

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0
                print(m)
            elif isinstance(m, LockedDropout) or isinstance(m, WeightDrop):
                m.dropout = 0
                print(m)
        batch = 0
        log.info("Starting calculation of Fisher's matrix")
        with tqdm(total=len(self.dataloader)) as pbar:
            for inputs, targets, lang, seq_len in self.dataloader:
                hidden = self.model.init_hidden(self.dataloader.batchsize)
                self.model.zero_grad()
                output, hidden, loss_typ = self.model(inputs, hidden, lang)

                loss = self.loss_function(self.model.decoder.weight, self.model.decoder.bias, output, targets)
                loss.backward()
                batch += 1

                pbar.update(1)

                for n, p in self.params.items():
                    precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: p / batch for n, p in precision_matrices.items()}
        log.info("Finished calculation of Fisher's matrix, it took", pbar.format_interval(pbar.format_dict['elapsed']))
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
