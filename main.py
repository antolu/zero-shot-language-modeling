#!/usr/bin/env python3

import logging
import math
import random
from datetime import datetime
from os import path
from pprint import pformat
import numpy as np

import torch
import tqdm
from torch.optim import Adam
from torch.nn.modules import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from laplace import GaussianPrior, LaplacePrior, VIPrior
from criterion import SplitCrossEntropyLoss
from data import get_sampling_probabilities, Dataset, Corpus, DataLoader
from engine import train, evaluate, refine
from models import RNN
from utils.parser import get_args
from utils import make_checkpoint, load_model
from regularisation import WeightDrop

timestamp = datetime.now().strftime('%Y%m%d_%H%M')

LOG_DIR = 'logs'
writer_dir = path.join(LOG_DIR, f'zerolm_{timestamp}')
tb_writer = SummaryWriter(writer_dir)

log = logging.getLogger()
log.setLevel(logging.DEBUG)

fh = logging.FileHandler(path.join(writer_dir, 'log.log'))
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

log.addHandler(fh)
log.addHandler(ch)


def main():
    args = get_args()

    log.info(f'Parsed arguments: \n{pformat(args.__dict__)}')
    assert args.cond_type.lower() in ['none', 'platanios', 'oestling']

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
            'languages': args.dev_langs + args.target_langs,
            'invert_include': True,
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

    if args.only_build_data:
        log.info('Finished building data tensors. Exiting')
        return

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

    if use_apex:
        optimizer = optimizers.FusedAdam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    else:
        params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_function.parameters())
        optimizer = Adam(params, lr=args.lr, weight_decay=args.wdecay)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

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

    # Add backward hook for gradient clipping
    if args.clip:
        if use_apex:
            for p in amp.master_params(optimizer):
                p.register_hook(lambda grad: torch.clamp(grad, -args.clip, args.clip))
        else:
            for p in model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -args.clip, args.clip))

    if args.prior == 'vi':
        prior = VIPrior(model, device=device)
        parameters['prior'] = prior

        def sample_weights(module: torch.nn.Module, input: torch.Tensor):
            prior.sample_weights(module)

        sample_weights_hook = model.register_forward_pre_hook(sample_weights)

    # Load model checkpoint if available
    start_epoch = 1
    if args.resume:
        if args.checkpoint is None:
            log.error('No checkpoint passed. Specify it using the --checkpoint flag')
            checkpoint = None
        else:
            log.info('Loading the checkpoint at {}'.format(args.checkpoint))
            checkpoint = load_model(args.checkpoint, **parameters)

            start_epoch = checkpoint['epoch']

        if args.wdrop:
            for rnn in model.rnns:
                if isinstance(rnn, WeightDrop):
                    rnn.dropout = args.wdrop
                elif rnn.zoneout > 0:
                    rnn.zoneout = args.wdrop

    saved_models = list()

    result_str = '| Language {} | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'

    def test():
        log.info('=' * 89)
        log.info('Running test set (zero-shot results)...')
        test_loss, avg_loss = evaluate(test_loader, **parameters)
        log.info('Test set finished | test loss {} | test bpc {}'.format(test_loss, test_loss / math.log(2)))

        tb_str = ''
        for lang, avg_l_loss in avg_loss.items():
            langstr = dictionary.idx2lang[lang]
            result = result_str.format(langstr, avg_l_loss, math.exp(avg_l_loss), avg_l_loss / math.log(2))
            log.info(result)
            tb_str += result + '\n'

            tb_writer.add_text('zero-shot results', tb_str)

        log.info('=' * 89)

    if args.train:
        f = 1.
        stored_loss = 1e32
        epochs_no_improve = 0

        val_losses = list()

        # calculate specific language lr
        data_spec_count = sum([len(ds) for l, ds in train_set.data.items()])
        data_spec_avg = data_spec_count / len(train_set.data.items())
        data_spec_lrweights = dict([(l, data_spec_avg / len(ds)) for l, ds in train_set.data.items()])

        # estimate total number of steps
        total_steps = sum([len(ds) // args.bptt for l, ds in train_set.data.items()]) * args.no_epochs
        steps = 0

        try:
            pbar = tqdm.trange(start_epoch, args.no_epochs + 1, position=1, dynamic_ncols=True)
            for epoch in pbar:

                steps = train(train_loader, lr_weights=data_spec_lrweights, **parameters,
                              total_steps=total_steps, steps=steps,
                              scaling=args.scaling, n_samples=args.n_samples, tb_writer=tb_writer)

                val_loss, _ = evaluate(val_loader, **parameters)
                pbar.set_description('Epoch {} | Val loss {}'.format(epoch, val_loss))
                log.info('End of epoch {} | Validation loss {}'.format(epoch, val_loss))

                # Save model
                if args.prior == 'vi':
                    sample_weights_hook.remove()

                filename = path.join(args.checkpoint_dir,
                                     '{}_epoch{}{}_{}.pth'.format(timestamp, epoch, '_with_apex' if use_apex else '',
                                     args.prior))
                torch.save(make_checkpoint(epoch + 1, **parameters), filename)
                saved_models.append(filename)

                if args.prior == 'vi':
                    sample_weights_hook = model.register_forward_pre_hook(sample_weights)

                # Early stopping
                if val_loss < stored_loss:
                    epochs_no_improve = 0
                    stored_loss = val_loss
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == args.patience:
                    log.info('Early stopping at epoch {}'.format(epoch))
                    log.info('Stored loss {} < new loss {}'.format(stored_loss, val_loss))
                    break

                val_losses.append(val_loss)

                # Reduce lr every 1/3 total epochs
                if epoch - 1 > f / 3 * args.no_epochs:
                    log.info('Epoch {}/{}. Dividing LR by 10'.format(epoch, args.no_epochs))
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] / 10

                    f += 1.
            test()
        except KeyboardInterrupt:
            log.info('Registered KeyboardInterrupt. Stopping training.')
            log.info('Saving last model to disk')

            if args.prior == 'vi':
                sample_weights_hook.remove()

            torch.save(make_checkpoint(epoch, **parameters),
                       path.join(args.checkpoint_dir,
                                 '{}_epoch{}{}_{}.pth'.format(timestamp, epoch, '_with_apex' if use_apex else '',
                                                              args.prior)))
            return
    elif args.test:
        test()

    # Only test on existing languages if there are no held out languages
    if not args.target_langs:
        exit(0)

    importance = 1e-5

    # If use UNIV, calculate informed prior, else use boring prior
    if args.prior == 'laplace':
        if not isinstance(prior, LaplacePrior):  # only calculate matrix if it is not supplied.
            log.info('Creating laplace approximation dataset')
            laplace_set = Dataset(data_splits['train'], batchsize=args.batchsize, bptt=100, reset_on_iter=True)
            laplace_loader = DataLoader(laplace_set, num_workers=args.workers)
            log.info('Creating Laplacian prior')
            prior = LaplacePrior(model, loss_function, laplace_loader, use_apex=use_apex, amp=amp,
                                               device=device)
            parameters['prior'] = prior

            torch.save(make_checkpoint('fisher_matrix', **parameters),
                       path.join(args.checkpoint_dir,
                                 '{}_fishers_matrix{}_{}.pth'.format(timestamp, '_with_apex' if use_apex else '',
                                                              args.prior)))
        importance = 1e5

    elif args.prior == 'ninf':
        log.info('Creating non-informative Gaussian prior')
        parameters['prior'] = GaussianPrior()
    elif args.prior == 'vi':
        importance = 1e-5
    elif args.prior == 'hmc':
        raise NotImplementedError
    else:
        raise ValueError(f'Passed prior {args.prior} is not an implemented inference technique.')

    best_model = saved_models[-1] if not len(saved_models) == 0 else args.checkpoint

    # Remove sampling hook from model
#    if args.prior == 'vi':
#        sample_weights_hook.remove()

    # Refine on 100 samples on each target
    if args.refine:
        # reset learning rate
        optimizer.param_groups[0]['lr'] = args.lr
        loss = 0

        results = dict()

        # Create individual tests sets
        test_sets = dict()
        for lang, lang_d in data_splits['test'].items():
            test_sets[lang] = DataLoader(
                Dataset({lang: lang_d}, make_config=True, batchsize=args.test_batchsize, bptt=args.bptt, eval=True),
                num_workers=args.workers)

        try:  # fail-safe in case cluster kills job
            for lang, lang_data in tqdm.tqdm(refine_set.items()):
                final_loss = False
                refine_dataloader = DataLoader(lang_data, num_workers=args.workers)
                load_model(best_model, **parameters)

                # reinstate dropout
                model.edrop.dropout = args.dropoute
                model.idrop.dropout = args.dropouti
                model.hdrop.dropout = args.dropouth
                model.odrop.dropout = args.dropout
                model.embedding_dropout.dropout = args.dropoute
                for rnn in model.rnns:
                    if isinstance(rnn, WeightDrop):
                        rnn.dropout = args.wdrop

                log.info(f'Refining for language {dictionary.idx2lang[lang]}')
                for epoch in range(1, args.refine_epochs + 1):
                    refine(refine_dataloader, **parameters, importance=importance)
                    if epoch % 5 == 0:
                        final_loss = True
                        loss, avg_loss = evaluate(test_sets[lang], model, loss_function, only_l=lang, report_all=True,
                                                  device=device)

                        for lang, avg_l_loss in avg_loss.items():
                            langstr = dictionary.idx2lang[lang]
                            log.debug(
                                result_str.format(langstr, avg_l_loss, math.exp(avg_l_loss), avg_l_loss / math.log(2)))

                if not final_loss:
                    loss, avg_loss = evaluate(test_sets[lang], model, loss_function, only_l=lang, report_all=True,
                                              device=device)

                for lang, avg_l_loss in avg_loss.items():
                    langstr = dictionary.idx2lang[lang]
                    log.info(result_str.format(langstr, avg_l_loss, math.exp(avg_l_loss), avg_l_loss / math.log(2)))
                    results[lang] = avg_l_loss

        except KeyboardInterrupt:
            log.error('Refine process stopped. Printing available results.')
        finally:
            log.info('=' * 89)
            log.info('FINAL FEW SHOT RESULTS: ')
            log.info('=' * 89)
            tb_str = ''
            for lang, avg_l_loss in results.items():
                langstr = dictionary.idx2lang[lang]
                result = result_str.format(langstr, avg_l_loss, math.exp(avg_l_loss), avg_l_loss / math.log(2))
                log.info(result)
                tb_str += result + '\n'
            log.info('=' * 89)

            tb_writer.add_text('few-shot results', tb_str)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception('Got exception from main process')

