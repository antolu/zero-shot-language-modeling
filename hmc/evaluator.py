import torch
from torch import nn

from data import DataLoader
from engine import Engine 

import logging

log = logging.getLogger(__name__)


class HMCEvaluator:
    def __init__(self, num_burn: int, engine: Engine, dataloader: DataLoader):
        self.round_counter = 0
        self.sum_w_sample = 0.0
        self.w_sample = 1.0

        self.num_burn = num_burn
        self.engine = engine
        self.dataloader = dataloader

        languages = dataloader.dataset.data.keys()
        self.average_log_likelihood = {l: 0 for l in languages}
        self.log_likelihoods = {l: [] for l in languages}

    def __call__(self, round_counter: int, *args, **kwargs):
        self.round_counter = round_counter
        alpha = self.__get_alpha()

        total_loss, avg_loss = self.engine.evaluate(self.dataloader)
        for language, loss in avg_loss.items():
            self.average_log_likelihood[language] *= (1 - alpha)
            self.average_log_likelihood[language] += alpha * loss
            self.log_likelihoods[language].append((round_counter, loss))

        log.debug(f'Time step {round_counter} | LL: {total_loss}')

    def __get_alpha(self):
        if self.round_counter < self.num_burn:
            return 1.0
        else:
            self.sum_w_sample += self.w_sample
            return self.w_sample / self.sum_w_sample
