from typing import Union

import torch
from torch import optim, nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from criterion import SplitCrossEntropyLoss
from data import DataLoader
from model import LSTM
from utils import detach


def train(dataloader: DataLoader, model: LSTM, optimizer: optim.optimizer,
          loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
          use_apex=False, amp=None, clip=None, parameters: list = None, alpha: float = 0., beta: float = 0., **kwargs):
    total_loss = 0
    hidden = model.init_hidden(dataloader.batchsize)

    batch = 0

    data_spec_count = sum([len(ds) for l, ds in dataloader.data.items()])
    data_spec_avg = data_spec_count / len(dataloader.data.items())
    data_spec_lrweights = dict([(l, data_spec_avg / len(ds)) for l, ds in dataloader.data.items()])

    with tqdm(total=len(dataloader)) as pbar:
        for data, targets, lang, seq_len in dataloader:

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / dataloader.bptt * data_spec_lrweights[lang]

            model.train()
            hidden = detach(hidden)
            optimizer.zero_grad()

            # TODO: use splitcrossentropy
            # raw_loss = loss_function(output, targets)
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

    dataloader.reset()


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

    with tqdm(total=len(dataloader)) as pbar:
        for data, targets, lang in dataloader:
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

    dataloader.reset()

    avg_loss = {lang: local_losses[lang].item() / len(dataloader.data[lang]) for lang in languages}
    total_loss = sum(avg_loss.values())

    return total_loss / len(languages)


def refine(lang, data_spec, model: LSTM, optimizer: torch.optim, bptt: int, clip: int, alpha: float = 0, beta: float = 0, importance: int = 100000):
    raise NotImplementedError
    batch, i = 0, 0
    total_loss = 0
    while i < data_spec.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(data_spec, i, args, seq_len=seq_len)
        hidden = model.init_hidden(eval_batch_size, lang)
        if args.model == 'QRNN':
            model.reset()

        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs, loss_typ = model(data, hidden, lang, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        laplace = importance * ewc.penalty(model)
        loss = raw_loss + laplace
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % 10 == 0 and batch > 0:
            cur_loss = total_loss.item() / 10
            elapsed = time.time() - start_time
            print('| {:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f} | laplace {}|'.format(
                batch, elapsed * 1000 / 100, cur_loss, math.exp(cur_loss), cur_loss / math.log(2), laplace.item()))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
