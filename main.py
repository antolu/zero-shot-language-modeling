#!/usr/bin/env python3

import torch
from ignite.engine import Engine, Events
from parser import get_args
import logging
from data import DataLoader, Dataset
import torch
import sys
sys.path.insert(0, 'lstm')

from torch.optim import Adam
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from lstm.model import RNNModel


log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


def main():
    args = get_args()

    loader = DataLoader(args.datadir)
    loader.load()

    return 0

    # if args.train:
    #     train_set = loader.get_split('train', 'eng')
    #     val_set = loader.get_split('valid', 'eng')
    # elif args.test:
    #     test_split = loader.get_split('test', 'eng')
    # else:
    #     raise NotImplementedError("Invalid options specified.")

    model = RNNModel('LSTM')

    optimizer = Adam(lr=1e-4)

    # TODO: implement universal prior
    # p(w|Dt) \propto exp[-.5 (w-w*)^T * H * (w-w*)],
    # H = -diag[f] + 1/sigma^2 * I
    # f_i = \sum^{l \in D_T} [\nabla log p(x^l | w) ]^2_i

    # trainer = create_supervised_evaluator(model, optimizer, criterion)


if __name__ == "__main__":
    main()
