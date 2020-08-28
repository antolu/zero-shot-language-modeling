#!/usr/bin/env python3

import logging
import math
import random
from datetime import datetime
from os import path
from pprint import pformat
import numpy as np
from bdb import BdbQuit

import torch
import tqdm
from torch.optim import Adam
from torch.nn.modules import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from criterion import SplitCrossEntropyLoss
from data import get_sampling_probabilities, Dataset, Corpus, DataLoader
from engine import train, evaluate, refine
from models import RNN
from utils.parser import get_args
from utils import make_checkpoint, load_model
from regularisation import WeightDrop
from hmc import HMC, apply_weights


log = logging.getLogger()
log.setLevel(logging.DEBUG)


def main():
    args = get_args()

    log.info(f'Parsed arguments: \n{pformat(args.__dict__)}')
    assert args.cond_type.lower() in ['none', 'platanios', 'oestling']


    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    writer_dir = path.join(args.logdir, f'zerolm_{timestamp}')
    tb_writer = SummaryWriter(writer_dir)

    fh = logging.FileHandler(path.join(writer_dir, 'log.log'))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(ch)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        log.info(f'Using device {torch.cuda.get_device_name()}')
    else:
        log.info('Using device cpu')

    use_apex = False
    if torch.cuda.is_available() and args.fp16:
        log.info('Loading Nvidia Apex and using AMP')
        from apex import amp, optimizers
        use_apex = True
    else:
        log.info('Using FP32')
        amp = None

    log.info(f'Using time stamp {timestamp} to save models and logs.')

    if not args.no_seed:
        log.info(f'Setting random seed to {args.seed} for reproducibility.')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    data = Corpus(args.datadir)

    data_splits = [
        {
            'split': 'train',
            'languages': args.dev_langs,
        },
        {
            'split': 'valid',
            'languages': args.dev_langs,
        },
        {
            'split': 'test',
            'languages': args.target_langs,
        },
    ]

    if args.refine:
        data_splits.append(
            {
                'split': 'train_100',
                'languages': args.target_langs,
                'ignore_missing': True,
            }
        )

    data_splits = data.make_datasets(data_splits, force_rebuild=args.rebuild)

    train_set, val_set, test_set = data_splits['train'], data_splits['valid'], data_splits['test']
    dictionary = data_splits['dictionary']

    train_language_distr = get_sampling_probabilities(train_set, args.lang_sampling_probs)
    train_set = Dataset(train_set, batchsize=args.batchsize, bptt=args.bptt, reset_on_iter=True,
                        language_probabilities=train_language_distr)
    val_set = Dataset(val_set, make_config=True, batchsize=args.valid_batchsize, bptt=args.bptt, eval=True)
    test_set = Dataset(test_set, make_config=True, batchsize=args.test_batchsize, bptt=args.bptt, eval=True)

    train_loader = DataLoader(train_set, num_workers=args.workers)
    val_loader = DataLoader(val_set, num_workers=args.workers)
    test_loader = DataLoader(test_set, num_workers=args.workers)

    if args.refine:
        refine_set = dict()
        for lang, lang_d in data_splits['train_100'].items():
            refine_set[lang] = Dataset({lang: lang_d}, batchsize=args.valid_batchsize, bptt=args.bptt, make_config=True)

    n_token = len(dictionary.idx2tkn)

    # Load and preprocess matrix of typological features
    # TODO: implement this, the OEST
    # prior_matrix = load_prior(args.prior, corpus.dictionary.lang2idx)
    # n_components = min(50, *prior_matrix.shape)
    # pca = PCA(n_components=n_components, whiten=True)
    # prior_matrix = pca.fit_transform(prior_matrix)
    prior = None

    if args.prior != 'vi':
        model = RNN(args.cond_type, prior, n_token, n_input=args.emsize, n_hidden=args.nhidden, n_layers=args.nlayers,
                    dropout=args.dropouto,
                    dropoute=args.dropoute, dropouth=args.dropouth, dropouti=args.dropouti, wdrop=args.wdrop,
                    wdrop_layers=[0, 1, 2], tie_weights=True).to(device)
    else:
        model = RNN(args.cond_type, prior, n_token, n_input=args.emsize, n_hidden=args.nhidden, n_layers=args.nlayers,
                    dropout=0,
                    dropoute=0, dropouth=0, dropouti=0, wdrop=0,
                    wdrop_layers=[0, 1, 2], tie_weights=True).to(device)

    if args.opt_level != 'O2' or not use_apex:  # Splitcross is not compatible with O2 optimization for amp
        loss_function = SplitCrossEntropyLoss(args.emsize, splits=[]).to(device)
    else:
        loss_function = CrossEntropyLoss().to(device)  # Should be ok to use with a vocabulary of this small size

    # Initialize optimizers, use FusedAdam if possible for speedup
    if use_apex:
        optimizer = optimizers.FusedAdam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    else:
        params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_function.parameters())
        optimizer = Adam(params, lr=args.lr, weight_decay=args.wdecay)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # Just for less cluttered parameter passing later
    parameters = {
        'model': model,
        'optimizer': optimizer,
        'loss_function': loss_function,
        'use_apex': use_apex,
        'amp': amp if use_apex else None,
        'clip': args.clip,
        'alpha': args.alpha,
        'beta': args.beta,
        'bptt': args.bptt,
        'device': device,
        'prior': args.prior,
    }

    saved_models = list()
    result_str = '| Language {} | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'

    def test():
        log.info('=' * 89)
        log.info('Running test set (zero-shot results)...')
        test_loss, avg_loss = evaluate(test_loader, **parameters)
        log.info('Test set finished | test loss {} | test bpc {}'.format(test_loss, test_loss / math.log(2)))

        for lang, avg_l_loss in avg_loss.items():
            langstr = dictionary.idx2lang[lang]
            result = result_str.format(langstr, avg_l_loss, math.exp(avg_l_loss), avg_l_loss / math.log(2))
            log.info(result)

        log.info('=' * 89)

    hmc = HMC(model=model, lr=args.lr, w_decay=args.wdecay, m_decay=0.01, num_burn=args.num_burn,
              w_decay_update=args.wdecay_update, use_apex=use_apex, amp=amp, device=device)

    average_results, all_results = hmc.sample(dataloader=train_loader, eval_dataloader=val_loader, **parameters, n_samples=args.n_samples, step_size=60)

    for lang, avg_l_loss in average_results.items():
        langstr = dictionary.idx2lang[lang]
        result = result_str.format(langstr, avg_l_loss, math.exp(avg_l_loss), avg_l_loss / math.log(2))
        log.info(result)

    for lang, res in all_results.items():
        for item in res:
            log.debug(item)


    # Only test on existing languages if there are no held out languages
    if not args.target_langs:
        exit(0)

    importance = 1e-5

    if args.importance != -1.0:
        log.info(f'Overriding importance {importance} with {args.importance}')
        importance = args.importance


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception('Got exception from main process')
