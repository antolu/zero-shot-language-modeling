import logging
import math
import time
from typing import Union
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import SplitCrossEntropyLoss
from laplace import Prior, VIPrior
from models.rnn import RNN
from utils import detach

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, model: RNN, optimizer: torch.optim.Optimizer,
                 criterion: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
                 use_apex: bool = False, amp=None,
                 prior: Union[str, Prior] = 'ninf',
                 scaling: str = None, n_samples: int = 4,
                 bptt: int = 125, alpha: float = 0., beta: float = 0., clip: float = 0.,
                 log_interval: int = 200, tb_writer=None,
                 device: Union[torch.device, str] = 'cpu',
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_apex = use_apex
        self.amp = amp
        self.prior = prior

        self.steps = 0

        # VI parameters
        self.scaling = scaling
        self.n_samples = n_samples

        # training parameters
        self.bptt = bptt
        self.alpha = alpha
        self.beta = beta

        # Add backward hook for gradient clipping
        if clip:
            if use_apex:
                for p in amp.master_params(optimizer):
                    p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
            else:
                for p in model.parameters():
                    p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))

        self.device = device

        # logging parameters
        self.log_interval = log_interval
        self.tb_writer = tb_writer

    def train(self, dataloader: DataLoader, lr_weights: dict = None, total_steps: int = 1, **kwargs):
        total_loss = 0
        i_batch = 0

        self.model.train()

        log.info('Starting training loop')
        start_time = time.time()

        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for batch in pbar:
                batch = tuple(t.squeeze(0).to(self.device) for t in batch)
                data, targets, seq_len, lang = batch

                hidden = self.model.init_hidden(batchsize=data.size(-1))

                lr2 = self.optimizer.param_groups[0]['lr']
                if lr_weights is not None:
                    self.optimizer.param_groups[0]['lr'] = lr2 * seq_len.item() / self.bptt * lr_weights[lang.item()]
                else:
                    self.optimizer.param_groups[0]['lr'] = lr2 * seq_len.item() / self.bptt

                hidden = detach(hidden)
                self.optimizer.zero_grad()

                loss = 0

                if not isinstance(self.prior, VIPrior):
                    n_samples = 1

                for s in range(n_samples):
                    output, hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, lang, return_h=True)

                    if isinstance(self.criterion, SplitCrossEntropyLoss) and isinstance(self.model, RNN):
                        raw_loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets)
                    else:
                        raw_loss = self.criterion(output, targets)

                    if self.alpha:
                        raw_loss = raw_loss + sum(
                            self.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                    # Temporal Activation Regularization (slowness)
                    if self.beta:
                        raw_loss = raw_loss + sum(self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

                    loss += raw_loss

                loss /= n_samples

                log_loss = loss

                if isinstance(self.prior, VIPrior):
                    kl_term = self.prior.kl_div()

                    if self.scaling == "uniform":
                        scale = 1. / total_steps
                    elif self.scaling == "linear_annealing":
                        scale = ((total_steps - self.steps - 1) * 2. + 1.) / total_steps ** 2
                    elif self.scaling == "logistic_annealing":
                        steepness = 0.0025
                        scale = 1. / (1 + np.exp(-steepness * (self.steps - total_steps / 2.)))
                    else:
                        scale = 1.
                    loss = loss + scale * kl_term

                if self.use_apex:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('train/loss', log_loss.item(), self.steps)

                    if isinstance(self.prior, VIPrior):
                        self.tb_writer.add_scalar('train/kl', kl_term.item(), self.steps)
                        self.tb_writer.add_scalar('train/loss+kl', loss.item(), self.steps)

                        if 'debug' in kwargs and kwargs['debug']:
                            if 'debug_streams' not in kwargs:
                                raise ValueError('debug streams not passed to train')

                            nts = self.prior.calculate_nts()

                            for n, val in nts.items():
                                kwargs['debug_streams'][n].write(
                                    ', '.join([str(elem) for elem in val.tolist()])
                                )

                self.optimizer.step()

                total_loss += raw_loss.data
                i_batch += 1
                self.steps += 1

                # reset lr to optimiser default
                self.optimizer.param_groups[0]['lr'] = lr2

                if i_batch % self.log_interval == 0 and batch > 0:
                    cur_loss = total_loss.item() / self.log_interval
                    elapsed = time.time() - start_time
                    log.debug(
                        '| {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                            i_batch, len(dataloader), self.optimizer.param_groups[0]['lr'], elapsed * 1000 / self.log_interval,
                            cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                    total_loss = 0
                    start_time = time.time()

                pbar.set_description('Training, end of batch {} | Loss {}'.format(batch, loss.data))

    def evaluate(self, dataloader: DataLoader, only_l: Union[torch.Tensor, int] = None, **kwargs):
        self.model.eval()

        languages = dataloader.dataset.data.keys()
        if only_l:
            if only_l not in languages:
                raise ValueError(f'Language {only_l} does not exist in the dataset')
            local_losses = {only_l: 0}
        else:
            local_losses = {lang: 0 for lang in languages}

        i_batch = 0
        prev_lang = ""

        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for batch in pbar:
                batch = tuple(t.squeeze(0).to(self.device) for t in batch)
                data, targets, seq_len, lang = batch

                if only_l and only_l != lang:
                    continue

                if prev_lang != lang:
                    prev_lang = lang
                    hidden = self.model.init_hidden(batchsize=data.size(-1))
                else:
                    detach(hidden)

                with torch.no_grad():
                    output, hidden = self.model(data, hidden, lang)
                    if isinstance(self.criterion, SplitCrossEntropyLoss) and isinstance(self.model, RNN):
                        loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets)
                    else:
                        loss = self.criterion(output, targets)
                    local_losses[lang.item()] += len(data) * loss.data

                i_batch += 1

                pbar.set_description('Evaluation, finished batch {} | loss {}'.format(i_batch, loss.data))

        avg_loss = {lang: local_losses[lang].item() / len(dataloader.dataset.data[lang]) for lang in
                    languages} if only_l is None else {
            only_l: local_losses[only_l].item() / len(dataloader.dataset.data[only_l])}
        total_loss = sum(avg_loss.values())

        return total_loss / len(languages), avg_loss

    def refine(self, dataloader: DataLoader, prior: Prior, importance: Union[int, float] = 100000, **kwargs):
        self.model.train()
        i_batch = 0
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for batch in pbar:
                batch = tuple(t.squeeze(0).to(self.device) for t in batch)
                data, targets, seq_len, lang = batch

                lr2 = self.optimizer.param_groups[0]['lr']
                self.optimizer.param_groups[0]['lr'] = lr2 * seq_len.item() / self.bptt
                hidden = self.model.init_hidden(batchsize=data.size(-1))
                self.optimizer.zero_grad()

                output, hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, lang, return_h=True)
                if isinstance(self.criterion, SplitCrossEntropyLoss) and isinstance(self.model, RNN):
                    loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets)
                else:
                    loss = self.criterion(output, targets)

                if isinstance(prior, Prior) and not isinstance(prior, VIPrior):
                    penalty = importance * prior.penalty(self.model)
                    loss += penalty
                else:
                    penalty = 'N/A'

                # Activiation Regularization
                if self.alpha:
                    loss = loss + sum(self.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                if self.beta:
                    loss = loss + sum(self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

                if self.use_apex:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()

                self.optimizer.param_groups[0]['lr'] = lr2

                i_batch += 1

                pbar.set_description(
                    'Loss {:5.2f} | bpc {:9.3f} | penalty {} |'.format(loss, loss / math.log(2), penalty.item() if isinstance(penalty, torch.Tensor) else penalty ))
