import torch
from torch import nn

from data import DataLoader
from engine import evaluate

import logging

log = logging.getLogger(__name__)


class HMCEvaluator:
    def __init__(self, num_burn: int, model: nn.Module, dataloader: DataLoader, criterion: torch.nn.modules.loss._Loss,
                 device: str = 'cpu'):
        self.round_counter = 0
        self.sum_w_sample = 0.0
        self.w_sample = 1.0

        self.num_burn = num_burn
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

        languages = dataloader.dataset.data.keys()
        self.average_log_likelihood = {l: 0 for l in languages}
        self.log_likelihoods = {l: [] for l in languages}

    def __call__(self, round_counter: int, *args, **kwargs):
        self.round_counter = round_counter
        alpha = self.__get_alpha()

        total_loss, avg_loss = evaluate(self.dataloader, self.model, self.criterion, device=self.device)
        for language, loss in avg_loss.items():
            self.average_log_likelihood[language] *= (1 - alpha)
            self.average_log_likelihood[language] += alpha * loss
            self.log_likelihoods[language].append(loss)

        log.debug(f'LL: {total_loss}')

    def __get_alpha(self):
        if self.round_counter < self.num_burn:
            return 1.0
        else:
            self.sum_w_sample += self.w_sample
            return self.w_sample / self.sum_w_sample
