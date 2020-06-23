import math
from typing import Union

import torch
from torch import optim, nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from bayesian import EWC
from criterion import SplitCrossEntropyLoss
from data import DataLoader
from model import LSTM
from utils import detach


def train(dataloader: DataLoader, model: LSTM, optimizer: optim.optimizer,
          loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
          use_apex=False, amp=None, clip=None, parameters: list = None, bptt: int = 125, alpha: float = 0.,
          beta: float = 0., **kwargs):

    total_loss = 0
    hidden = model.init_hidden(dataloader.batchsize)

    batch = 0

    data_spec_count = sum([len(ds) for l, ds in dataloader.data.items()])
    data_spec_avg = data_spec_count / len(dataloader.data.items())
    data_spec_lrweights = dict([(l, data_spec_avg / len(ds)) for l, ds in dataloader.data.items()])

    with tqdm(dataloader) as pbar:
        for data, targets, seq_len, lang in pbar:

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / bptt * data_spec_lrweights[lang]

            model.train()
            hidden = detach(hidden)
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs, loss_typ = model(data, hidden, lang, return_h=True)
            if isinstance(loss_function, SplitCrossEntropyLoss):
                raw_loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
            else:
                raw_loss = loss_function(output, targets)
            loss = raw_loss

            if alpha:
                loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if beta:
                loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            if loss_typ:
                loss = loss + (1e-2 * loss_typ)

            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if clip:
                nn.utils.clip_grad_norm_(parameters, clip)

            optimizer.step()

            total_loss += raw_loss.data
            batch += 1

            # reset lr to optimiser default
            optimizer.param_groups[0]['lr'] = lr2

            pbar.set_description('Training, end of batch {} | Loss {}'.format(batch, loss.data))
            pbar.update(1)


def evaluate(dataloader: DataLoader, model: LSTM, loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
             only_l: str = None, **kwargs):
    model.eval()

    languages = dataloader.data.keys()
    if only_l:
        if only_l not in languages:
            raise ValueError(f'Language {only_l} does not exist in the dataset')
        local_losses = {only_l: 0}
    else:
        local_losses = {lang: 0 for lang in languages}

    hidden = model.init_hidden(dataloader.batchsize)

    batch = 0

    with tqdm(dataloader) as pbar:
        for data, targets, seq_len, lang in pbar:
            if only_l and only_l != lang:
                continue

            output, hidden = model(data, hidden, lang)
            if isinstance(loss_function, SplitCrossEntropyLoss):
                loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
            else:
                loss = loss_function(output, targets)
            local_losses[lang] += len(data) * loss.data
            hidden = detach(hidden)

            batch += 1

            pbar.set_description('Evaluation, finished batch {} | loss {}'.format(batch, loss.data))

    avg_loss = {lang: local_losses[lang].item() / len(dataloader.data[lang]) for lang in languages}
    total_loss = sum(avg_loss.values())

    return total_loss / len(languages)


def refine(dataloader: DataLoader, model: LSTM, optimizer: torch.optim,
           loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss], ewc: EWC, bptt: int, clip: int,
           parameters: list, use_apex: bool = False, amp=None, alpha: float = 0, beta: float = 0,
           importance: int = 100000):
    with tqdm(dataloader) as pbar:
        for data, targets, lang, seq_len in pbar:

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / bptt
            model.train()
            hidden = model.init_hidden(dataloader.batchsize)
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs, loss_typ = model(data, hidden, lang, return_h=True)
            if isinstance(loss_function, SplitCrossEntropyLoss):
                loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
            else:
                loss = loss_function(output, targets)

            laplace = importance * ewc.penalty(model)
            loss += laplace

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

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if clip:
                torch.nn.utils.clip_grad_norm_(parameters, clip)

            optimizer.step()

            optimizer.param_groups[0]['lr'] = lr2

            pbar.set_description(
                'Loss {:5.2f} | bpc {:9.3f} | laplace {} |'.format(loss, loss / math.log(2), laplace.item()))
