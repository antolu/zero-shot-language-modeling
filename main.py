#!/usr/bin/env python3

import logging
import math
import random
import warnings
from datetime import datetime
from os import path
from pprint import pformat

import torch
import tqdm
from torch.optim import Adam

from bayesian import BoringPrior, EWC
from criterion import SplitCrossEntropyLoss
from data import Data, DataLoader, get_sampling_probabilities
from engine import train, evaluate
from model import LSTM
from parser import get_args
from utils import get_checkpoint, save_model, load_model, DotDict

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


def main():
    args = get_args()

    log.info(f'Parsed arguments: \n{pformat(args.__dict__)}')
    assert args.cond_type.lower() in ['none', 'platanios', 'oestling']

    use_apex = False
    if torch.cuda.is_available() and args.fp16:
        log.info('Loading Nvidia Apex and using AMP')
        from apex import amp
        use_apex = True
    else:
        log.info('Using FP32')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info('Using device {}.'.format(device))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log.info(f'Using time stamp {timestamp} to save models.')

    if not args.no_seed:
        log.info(f'Setting random seed to {args.seed} for reproducibility.')
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    data = Data(args.datadir, args.dataset)
    data.load(args.rebuild)

    # TODO: only load data tensors upon calling get_split, else only save the paths in memory
    train_set = data.get_split('train')
    val_set = data.get_split('valid')  # TODO: use args.dev_lang to get validation set
    test_set = data.get_split('test')

    train_language_distr = get_sampling_probabilities(train_set, 0.8)
    train_loader = DataLoader(train_set, args.batchsize, bptt=args.bptt,
                              lang_sampler=train_language_distr, device=device, eval=False)
    val_loader = DataLoader(val_set, args.valid_batchsize, device=device, eval=True)
    test_loader = DataLoader(test_set, args.test_batchsize, device=device, eval=True)

    n_token = len(data.idx_to_character)

    # Load and preprocess matrix of typological features
    # TODO: implement this, the OEST
    # prior_matrix = load_prior(args.prior, corpus.dictionary.lang2idx)
    # n_components = min(50, *prior_matrix.shape)
    # pca = PCA(n_components=n_components, whiten=True)
    # prior_matrix = pca.fit_transform(prior_matrix)
    prior = None

    model = LSTM(args.cond_type, prior, n_token, n_input=args.emsize, n_hidden=args.nhidden, n_layers=args.nlayers,
                 dropout=args.dropout,
                 dropoute=args.dropoute, dropouth=args.dropouth, dropouti=args.dropouti, wdrop=args.wdrop,
                 wdrop_layers=[0], tie_weights=args.tie_weights).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    loss_function = SplitCrossEntropyLoss(args.emsize, splits=[]).to(device)
    # loss_function = nn.CrossEntropyLoss().to(device)  # Should be ok to use with a dataset of this small size
    params = list(model.parameters()) + list(loss_function.parameters())

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer)

    parameters = DotDict({
        'model': model,
        'optimizer': optimizer,
        'loss': loss_function,
        'parameters': params,
        'use_apex': use_apex,
        'amp': amp if use_apex else None,
        'clip': args.clip,
        'alpha': args.alpha,
        'beta': args.beta,
    })

    # Load model checkpoint if available
    start_epoch = 1
    if args.resume:
        if args.checkpoint is None:
            log.error('No checkpoint passed. Specify it using the --checkpoint flag')
            checkpoint = None
        else:
            log.info('Loading the checkpoint at {}'.format(args.checkpoint))
            checkpoint = load_model(args.checkpoint, **parameters)

            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            loss_function.load_state_dict(checkpoint['loss'])

            start_epoch = checkpoint['epoch']

            if use_apex:
                amp.load_state_dict(checkpoint['amp'])

    saved_models = list()

    def test():
        log.info('-' * 89)
        log.info('Running test set...')
        test_loss = evaluate(test_loader, **parameters)
        log.info('Test set finished | test loss {} | test bpc {}'.format(test_loss, test_loss / math.log(2)))
        log.info('-' * 89)

    if args.train:
        f = 1.
        stored_loss = 1e32
        epochs_no_improve = 0

        val_losses = list()

        try:
            pbar = tqdm.trange(start_epoch, args.no_epochs + 1)
            for epoch in pbar:
                train(train_loader, **parameters)

                val_loss = evaluate(val_loader, **parameters)
                pbar.set_description('Epoch {} | Val loss {}'.format(epoch, val_loss))

                # Save model
                filename = path.join(args.dir_model,
                                     '{}_epoch{}{}.pth'.format(timestamp, epoch, '_with_apex' if use_apex else ''))
                save_model(filename, get_checkpoint(epoch + 1, **parameters))
                saved_models.append(filename)

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

            save_model(path.join(args.dir_model,
                                 '{}_epoch{}{}.pth'.format(timestamp, epoch, '_with apex' if use_apex else '')),
                       get_checkpoint(epoch))
    elif args.test:
        test()

    # Only test on existing languages if there are no held out languages
    if not args.target:
        exit(0)

    # If use UNIV, calculate informed prior, else use boring prior
    if args.laplace:
        laplace_loader = DataLoader(train_set, args.batchsize, bptt=100, device=device, eval=False)
        ewc = EWC(model, loss_function, laplace_loader)
    else:
        ewc = BoringPrior()

    # Refine on 100 samples on each target
    if args.refine:
        # TODO: implement laplacian approximation
        raise NotImplementedError


if __name__ == "__main__":
    main()
