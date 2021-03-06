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


def train(dataloader: DataLoader, model: RNN, optimizer: torch.optim.Optimizer,
          loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss], use_apex=False, amp=None,
          lr_weights: dict = None, prior: str = 'ninf', scaling: str = None, total_steps: int = 0, steps: int = 0,
          bptt: int = 125, alpha: float = 0., beta: float = 0., log_interval: int = 200, n_samples: int = 4,
          device: Union[torch.device, str] = 'cpu', tb_writer=None, **kwargs):
    total_loss = 0
    batch = 0

    tr_kl = 0.
    logging_kl = 0.
    tr_loss = 0.
    logging_loss = 0.

    model.train()

    log.info('Starting training loop')
    start_time = time.time()

    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for data, targets, seq_len, lang in pbar:

            data = data.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)
            lang = lang.to(device)

            hidden = model.init_hidden(batchsize=data.size(-1))

            lr2 = optimizer.param_groups[0]['lr']
            if lr_weights is not None:
                optimizer.param_groups[0]['lr'] = lr2 * seq_len.item() / bptt * lr_weights[lang.item()]
            else:
                optimizer.param_groups[0]['lr'] = lr2 * seq_len.item() / bptt

            hidden = detach(hidden)
            optimizer.zero_grad()

            loss = 0

            if not isinstance(prior, VIPrior):
                n_samples = 1

            for s in range(n_samples):
                output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, lang, return_h=True)

                if isinstance(loss_function, SplitCrossEntropyLoss):
                    raw_loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
                else:
                    raw_loss = loss_function(output, targets)

                if alpha:
                    raw_loss = raw_loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                if beta:
                    raw_loss = raw_loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

                loss += raw_loss

            loss /= n_samples

            log_loss = loss

            if isinstance(prior, VIPrior):
                kl_term = prior.kl_div()

                if scaling == "uniform":
                    scale = 1. / total_steps
                elif scaling == "linear_annealing":
                    scale = ((total_steps - steps - 1) * 2. + 1.) / total_steps ** 2
                elif scaling == "logistic_annealing":
                    steepness = 0.0025
                    scale = 1. / (1 + np.exp(-steepness * (steps - total_steps / 2.)))
                else:
                    scale = 1.
                loss = loss + scale * kl_term
                tr_kl += kl_term.item()

            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', log_loss.item(), steps)

                if isinstance(prior, VIPrior):
                    tb_writer.add_scalar('train/kl', kl_term.item(), steps)
                    tb_writer.add_scalar('train/loss+kl', loss.item(), steps)

                    logging_kl += tr_kl

                logging_loss += tr_loss

            optimizer.step()

            total_loss += raw_loss.data
            batch += 1
            steps += 1

            # reset lr to optimiser default
            optimizer.param_groups[0]['lr'] = lr2

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / log_interval
                elapsed = time.time() - start_time
                log.debug(
                    '| {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        batch, len(dataloader), optimizer.param_groups[0]['lr'], elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()

            pbar.set_description('Training, end of batch {} | Loss {}'.format(batch, loss.data))

    return steps


def evaluate(dataloader: DataLoader, model: RNN, loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
             only_l: Union[torch.Tensor, int] = None, device: Union[torch.device, str] = 'cpu', **kwargs):
    model.eval()

    languages = dataloader.dataset.data.keys()
    if only_l:
        if only_l not in languages:
            raise ValueError(f'Language {only_l} does not exist in the dataset')
        local_losses = {only_l: 0}
    else:
        local_losses = {lang: 0 for lang in languages}

    batch = 0
    prev_lang = ""

    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for data, targets, seq_len, lang in pbar:
            data = data.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)
            lang = lang.to(device)

            if only_l and only_l != lang:
                continue

            if prev_lang != lang:
                prev_lang = lang
                hidden = model.init_hidden(batchsize=data.size(-1))
            else:
                detach(hidden)

            with torch.no_grad():
                output, hidden = model(data, hidden, lang)
                if isinstance(loss_function, SplitCrossEntropyLoss):
                    loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
                else:
                    loss = loss_function(output, targets)
                local_losses[lang.item()] += len(data) * loss.data

            batch += 1

            pbar.set_description('Evaluation, finished batch {} | loss {}'.format(batch, loss.data))

    avg_loss = {lang: local_losses[lang].item() / len(dataloader.dataset.data[lang]) for lang in languages} if only_l is None else {only_l: local_losses[only_l].item() / len(dataloader.dataset.data[only_l])}
    total_loss = sum(avg_loss.values())

    return total_loss / len(languages), avg_loss


def refine(dataloader: DataLoader, model: RNN, optimizer: torch.optim.Optimizer,
           loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss], prior: Prior, bptt: int,
           use_apex: bool = False,
           amp=None, alpha: float = 0, beta: float = 0, importance: Union[int, float] = 100000,
           device: Union[torch.device, str] = 'cpu', **kwargs):
    model.train()
    batch = 0
    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for data, targets, seq_len, lang in pbar:
            data = data.squeeze(0).to(device)
            targets = targets.squeeze(0).to(device)
            lang = lang.to(device)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len.item() / bptt
            hidden = model.init_hidden(batchsize=data.size(-1))
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, lang, return_h=True)
            if isinstance(loss_function, SplitCrossEntropyLoss):
                loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
            else:
                loss = loss_function(output, targets)

            penalty = importance * prior.penalty(model)
            loss += penalty

            # Activiation Regularization
            if alpha:
                loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if beta:
                loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            optimizer.param_groups[0]['lr'] = lr2

            batch += 1

            pbar.set_description(
                'Loss {:5.2f} | bpc {:9.3f} | penalty {} |'.format(loss, loss / math.log(2), penalty.item()))
