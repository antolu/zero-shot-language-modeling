#!/usr/bin/env python3

import logging
import math
import sys
import traceback
from datetime import datetime
from os import path

import torch
import torch.nn as nn
import tqdm
from torch.distributions.normal import Normal

from data import Data, DataLoader, get_sampling_probabilities
from parser import get_args
from utils import save_model, load_model, detach

sys.path.insert(0, 'lstm')

from torch.optim import Adam
# from model import LSTM
from lstm.model import RNNModel

log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


# TODO: softmax for loss
# TODO: variational dropout for model
# TODO: implement universal prior


def main():
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info('Using device {}.'.format(device))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    data = Data(args.datadir, args.dataset)
    data.load(args.remap, args.rebuild)

    train_set = data.get_split('train')
    val_set = data.get_split('valid')
    test_set = data.get_split('test')

    train_language_distr = get_sampling_probabilities(train_set, 0.8)
    seq_len_distr = Normal(loc=args.bptt, scale=5)
    train_loader = DataLoader(train_set, args.batchsize, seq_len=seq_len_distr, lang_sampler=train_language_distr,
                              device=device, eval=False)
    val_loader = DataLoader(val_set, args.valid_batchsize, device=device, eval=True)
    test_loader = DataLoader(test_set, args.test_batchsize, device=device, eval=True)

    n_token = len(data.idx_to_character)

    # model = LSTM(400, n_hidden=1150, n_layers=3)
    model = RNNModel('LSTM', ntoken=n_token, nhid=1840, ninp=400, nlayers=3, dropout=args.dropout,
                     dropoute=args.dropoute, dropouth=args.dropouth, dropouti=args.dropouti, wdrop=args.wdrop)
    # model = model.to(device)

    # backwards combatibility
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss().to(device)
    params = list(model.parameters()) + list(loss_function.parameters())

    # TODO: implement universal prior
    # p(w|Dt) \propto exp[-.5 (w-w*)^T * H * (w-w*)],
    # H = -diag[f] + 1/sigma^2 * I
    # f_i = \sum^{l \in D_T} [\nabla log p(x^l | w) ]^2_i

    def evaluate(dataloader: DataLoader):
        model.eval()

        total_loss = 0

        hidden = model.init_hidden(dataloader.batchsize)

        total_iters = dataloader.get_total_iters(args.bptt)
        batch = 0

        with tqdm.tqdm(total=total_iters) as pbar:
            while not dataloader.is_exhausted():
                data, targets = dataloader.get_batch()
                output, hidden = model(data, hidden)
                loss = loss_function(output, targets)
                total_loss += len(data) * loss.data
                hidden = detach(hidden)

                batch += 1

                pbar.set_description('Evaluation, finished batch {} | loss {}'.format(batch, loss.data))

        dataloader.reset()

        return total_loss / args.test_batchsize

    def train(dataloader: DataLoader):
        total_loss = 0
        hidden = model.init_hidden(dataloader.batchsize)

        batch = 0

        total_iters = dataloader.get_total_iters(args.bptt)

        with tqdm.tqdm(total=total_iters) as pbar:
            while not dataloader.is_exhausted():
                data, targets, seq_len = dataloader.get_batch()

                lr2 = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt

                model.train()
                hidden = detach(hidden)
                optimizer.zero_grad()

                output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
                raw_loss = loss_function(output, targets)

                loss = raw_loss

                if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)

                loss.backward()
                optimizer.step()

                total_loss += raw_loss.data
                batch += 1

                # reset lr to optimiser default
                optimizer.param_groups[0]['lr'] = lr2

                pbar.set_description('Training, end of batch {} | Loss {}'.format(batch, loss.data))
                pbar.update(1)

        dataloader.reset()

    def test():
        log.info('-' * 89)
        log.info('Running test set...')
        test_loss = evaluate(test_loader)
        log.info('Test set finished | test loss {} | test bpc {}'.format(test_loss, test_loss / math.log(2)))
        log.info('-' * 89)

    if args.train:
        f = 1.
        stored_loss = 1e32
        epochs_no_improve = 0

        val_losses = list()

        try:
            pbar = tqdm.trange(1, args.no_epochs)
            for epoch in pbar:
                train(train_loader)

                val_loss = evaluate(val_set)
                pbar.set_description('Epoch {} | Val loss {}'.format(epoch, val_loss))

                # Save model
                save_model(path.join(args.dir_model, '{}_epoch{}.pth'.format(timestamp, epoch)),
                           (model, loss_function, optimizer))

                # Early stopping
                if val_loss < stored_loss:
                    epochs_no_improve = 0
                    stored_loss = val_loss
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == args.patience:
                    log.info('Early stopping at epoch {}'.format(epoch))
                    break

                val_losses.append(val_loss)

                # Reduce lr every 1/3 total epochs
                if epoch > f / 3 * args.no_epochs:
                    log.info('Epoch {}/{}. Dividing LR by 10'.format(epoch, args.no_epochs))
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10
                    f += 1.
            test()
        except KeyboardInterrupt:
            log.info('Registered KeyboardInterrupt. Stopping training.')
            log.info('Saving last model to disk')
            save_model(path.join(args.dir_model, '{}_epoch{}.pth'.format(timestamp, epoch)),
                       (model, loss_function, optimizer))
    else:
        test()


if __name__ == "__main__":
    main()
