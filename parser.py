from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    # Arguments concerning the environment of the repository
    parser.add_argument("--datadir", type=str, default="dataset",
                        help="Path to the root directory containing the datasets.")
    parser.add_argument("--dataset", type=str, default="latin",
                        help="Which dataset to use", choices=['ipa', 'latin'])
    parser.add_argument('--dir_model', default='checkpoints/',
                        help='Path to project data')
    parser.add_argument('--checkpoint', '-c', default=None,
                        help='path to pretrained-model')
    parser.add_argument('--remap', action='store_true', help='Redo character mappings.')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the data vectors.')

    # Arguments concerning training and testing the model
    traintest = parser.add_mutually_exclusive_group(required=True)
    traintest.add_argument("--train", action="store_true",
                           help="Train the model")
    traintest.add_argument("--test", action="store_true",
                           help="Test the model")
    parser.add_argument('--resume', action='store_true', help='Resume training', default=False)
    parser.add_argument("--no-epochs", type=int, default=6, dest="no_epochs",
                        help="Number of epochs to train the model")
    parser.add_argument("-lr", "--lr", type=float, default=1.5e-4,
                        help="The learning rate for the Adam optimiser.")
    parser.add_argument('--bptt', type=float, default=125, help='Mean sequence length')
    parser.add_argument('--clip', type=float, default=0.25, help='Gradient clipping')
    parser.add_argument("--batchsize", '--train-batchsize', type=int, default=128,
                        help="The batchsize to use in training")
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout applied to layers')
    parser.add_argument('--dropouti', type=float, default=0.1, help='Dropout for input embedding layers')
    parser.add_argument('--dropouth', type=float, default=0.1, help='Dropout for rnn layers')
    parser.add_argument('--dropoute', type=float, default=0.1, help='Dropout to remove words from embedding layer')
    parser.add_argument('--wdrop', type=float, default=0.2,
                        help='Amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--valid-batchsize', type=int, default=10, dest='valid_batchsize')
    parser.add_argument('--test-batchsize', type=int, default=1, dest='test_batchsize')
    parser.add_argument('--embedding-size', type=int, default=400, help='The size of the embedding.')
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="The difference between losses between iterations \
                        to break.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 or FP32 in training.")
    parser.add_argument("--opt-level", dest="opt_level", type=str, default="O1",
                        help="Which optimisation to use for mixed precision training.")
    parser.add_argument("--eval-metric", type=str, dest="eval_metric", default="accuracy",
                        choices=["accuracy", "attention"])

    parser.add_argument('--workers', default=8, type=int,
                        help="Number of workers for training the network")
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--start-epoch', default=0, type=int, dest="start_epoch",
                        help="The epoch to start/resume training at")
    parser.add_argument('--patience', default=1, type=int)

    args = parser.parse_args()

    return args
