#!/usr/bin/env python3

import torch
from ignite.engine import Engine, Events
from parser import get_args
import logging
from data import DataLoader, Dataset
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'lstm')

from torch.optim import Adam
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
# from model import LSTM
from lstm.model import RNNModel


log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

# TODO: Save torch models to disk to save time
# TODO: Batch size splitting
# TODO: gradient clipping
# TODO: implement universal prior


def main():
    args = get_args()

    loader = DataLoader(args.datadir, args.dataset)
    loader.load(args.remap, args.rebuild)
    loader.make_batches(128)

    n_token = len(loader.idx_to_character)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # model = LSTM(400, n_hidden=1150, n_layers=3)
    model = RNNModel('LSTM', ntoken=n_token, nhid=1150, ninp=400, nlayers=3).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)

    loss_function = nn.CrossEntropyLoss()

    eval_batchsize = 10
    test_batchsize = 1

    return


    # TODO: implement universal prior
    # p(w|Dt) \propto exp[-.5 (w-w*)^T * H * (w-w*)],
    # H = -diag[f] + 1/sigma^2 * I
    # f_i = \sum^{l \in D_T} [\nabla log p(x^l | w) ]^2_i

    trainer = create_supervised_trainer(model, optimizer, loss_function)
    evaluator = create_supervised_evaluator(model, metrics={'loss' : nn.NLLLoss()})

    # @trainer.on(Events.EPOCH_STARTED)

    @trainer.on(Events.ITERATION_COMPLETED)
    def validate(trainer):
        evaluator.run(val_set)
        metrics = evaluator.state.metrics

        # early stopping

    trainer.run(train_set, max_epochs=args.no_epochs)


if __name__ == "__main__":
    main()
