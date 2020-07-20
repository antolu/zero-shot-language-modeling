from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    # Arguments concerning the environment of the repository
    parser.add_argument('--datadir', type=str, default='dataset/bibles_latin',
                        help='Path to the root directory containing the datasets.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', dest='checkpoint_dir', help='The directory to save training checkpoints to')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to pretrained-model')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the data vectors.')

    # Arguments concerning training and testing the model
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--refine', action='store_true', help='Refine the model using laplacian approximation')

    parser.add_argument('--prior', choices=['ninf', 'laplace', 'vi', 'hmc'], default='ninf',
                        help='Which technique to use for inference of the universal prior')

    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--start-epoch', default=1, type=int, dest='start_epoch',
                        help='The epoch to start/resume training at')
    parser.add_argument('--no-epochs', type=int, default=6, dest='no_epochs',
                        help='Number of epochs to train the model')
    parser.add_argument('--refine-epochs', type=int, default=25, dest='refine_epochs', help='Number of epochs to refine model with fisher matrix')
    parser.add_argument('-lr', '--lr', type=float, default=1.e-4, help='The learning rate for the Adam optimiser.')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='Weight decay applied to all weights')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--bptt', type=float, default=125, help='Mean sequence length (backprop through time')
    parser.add_argument('--clip', type=float, default=0.25, help='Amount of gradient clipping')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='Beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--no-seed', action='store_true', dest='no_seed', help='Do not set a random seed.')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')

    # Arguments concerning the model
    parser.add_argument('--cond-type', type=str, dest='cond_type', default='None',
                        choices=['none', 'platanios', 'sutskever', 'oestling'],
                        help='Which condFrais de dossiers : compris dans la commissionition type to use for training the model.')
    parser.add_argument('--nhidden', type=int, default=1150, help='Number of hidden units in the LSTM.')
    parser.add_argument('--emsize', type=int, default=400, help='The size of the embeddings in the LSTM.')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of layers in the LSTM')

    # Regularisation options
    parser.add_argument('--dropouto', type=float, default=0.4, help='Dropout for output hidden layers')
    parser.add_argument('--dropouti', type=float, default=0.1, help='Dropout for input embedding layers')
    parser.add_argument('--dropouth', type=float, default=0.1, help='Dropout for intermediate hidden layers')
    parser.add_argument('--dropoute', type=float, default=0.1, help='Dropout to remove words from embedding layer')
    parser.add_argument('--wdrop', type=float, default=0.2,
                        help='Amount of weight dropout to apply to the RNN hidden to hidden matrix')

    # Data loading options
    parser.add_argument('--batchsize', '--train-batchsize', type=int, default=128,
                        help='The batchsize to use in training')
    parser.add_argument('--valid-batchsize', type=int, default=10, dest='valid_batchsize')
    parser.add_argument('--test-batchsize', type=int, default=1, dest='test_batchsize')
    parser.add_argument('--dev-langs', dest='dev_langs', nargs='+', default=[],
                        help='Target languages on which performing zero- or few-shot evaluation')
    parser.add_argument('--target-langs', dest='target_langs', nargs='+', default=[],
                        help='Target languages on which performing zero- or few-shot evaluation')

    # Mixed precision settings
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use FP16 or FP32 in training.')
    parser.add_argument('--opt-level', dest='opt_level', type=str, default='O1',
                        help='Which optimisation to use for mixed precision training.')

    # Not used currently. Might get implemented for parallelized data loading
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of workers for training the network')

    # Early stopping arguments
    parser.add_argument('--patience', default=1, type=int,
                        help='How many epochs to wait for loss improvement, else stop early.')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='The difference between losses between iterations \
                            to break.')

    args = parser.parse_args()

    return args
