import math
import time
from typing import Union

import logging
import torch
from torch import optim, nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from bayesian import EWC
from criterion import SplitCrossEntropyLoss
from data import DataLoader
from model import LSTM
from utils import detach

log = logging.getLogger('zerolm')


def train(dataloader: DataLoader, model: LSTM, optimizer: torch.optim.Optimizer,
          loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
          use_apex=False, amp=None, clip=None, parameters: list = None, bptt: int = 125, alpha: float = 0.,
          beta: float = 0., log_interval: int = 200, **kwargs):
    total_loss = 0

    batch = 0

    data_spec_count = sum([len(ds) for l, ds in dataloader.data.items()])
    data_spec_avg = data_spec_count / len(dataloader.data.items())
    data_spec_lrweights = dict([(l, data_spec_avg / len(ds)) for l, ds in dataloader.data.items()])

    log.info('Starting training loop')
    start_time = time.time()

    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        for data, targets, seq_len, lang in pbar:
            hidden = model.init_hidden(dataloader.batchsize)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / bptt * data_spec_lrweights[lang]

            model.train()
            hidden = detach(hidden)
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, lang, return_h=True)
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

            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            total_loss += raw_loss.data
            batch += 1

            # reset lr to optimiser default
            optimizer.param_groups[0]['lr'] = lr2

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / log_interval
                elapsed = time.time() - start_time
                log.debug(
                    '| {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        batch, data_spec_count // bptt, optimizer.param_groups[0]['lr'], elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                total_loss = 0
                start_time = time.time()

            pbar.set_description('Training, end of batch {} | Loss {}'.format(batch, loss.data))
            pbar.update(1)


def evaluate(dataloader: DataLoader, model: LSTM, loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
             only_l: str = None, report_all: bool = False, **kwargs):
    model.eval()

    languages = dataloader.data.keys()
    if only_l:
        if only_l not in languages:
            raise ValueError(f'Language {only_l} does not exist in the dataset')
        local_losses = {only_l: 0}
    else:
        local_losses = {lang: 0 for lang in languages}

    batch = 0
    prev_lang = ""

    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        for data, targets, seq_len, lang in pbar:
            if only_l and only_l != lang:
                continue

            if prev_lang != lang:
                prev_lang = lang
                hidden = model.init_hidden(dataloader.batchsize)
            else:
                detach(hidden)

            with torch.no_grad():
                output, hidden = model(data, hidden, lang)
                if isinstance(loss_function, SplitCrossEntropyLoss):
                    loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
                else:
                    loss = loss_function(output, targets)
                local_losses[lang] += len(data) * loss.data

            batch += 1

            pbar.set_description('Evaluation, finished batch {} | loss {}'.format(batch, loss.data))

    avg_loss = {lang: local_losses[lang].item() / len(dataloader.data[lang]) for lang in languages}
    total_loss = sum(avg_loss.values())

    if report_all:
        langstr = corpus.dictionary.idx2lang[lang.item()]
        print('=' * 89)
        print('| Language {} | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            langstr, avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))

    return total_loss / len(languages)


def refine(dataloader: DataLoader, model: LSTM, optimizer: torch.optim.Optimizer,
           loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss], ewc: EWC, bptt: int, clip: int,
           parameters: list, use_apex: bool = False, amp=None, alpha: float = 0, beta: float = 0,
           importance: int = 100000):

    model.train()
    batch = 0
    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        for data, targets, seq_len, lang in pbar:

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / bptt
            hidden = model.init_hidden(dataloader.batchsize)
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, lang, return_h=True)
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

            optimizer.step()

            optimizer.param_groups[0]['lr'] = lr2

            batch += 1

            pbar.set_description(
                'Loss {:5.2f} | bpc {:9.3f} | laplace {} |'.format(loss, loss / math.log(2), laplace.item()))
