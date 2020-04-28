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
    parser.add_argument('--pretrained-model', dest='pretrained_model', default=None,
                        help='path to pretrained-model')
    parser.add_argument('--remap', action='store_true', help='Redo character mappings.')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the data vectors.')

    # Arguments concerning training and testing the model
    traintest = parser.add_mutually_exclusive_group(required=True)
    traintest.add_argument("--train", action="store_true",
                           help="Train the model")
    traintest.add_argument("--test", action="store_true",
                           help="Test the model")
    parser.add_argument("--no-epochs", type=int, default=100, dest="no_epochs",
                        help="Number of epochs to train the model")
    parser.add_argument("-lr", "--lr", type=float, default=1.5e-4,
                        help="The learning rate for the Adam optimiser.")
    parser.add_argument("--batchsize", type=int, default=80,
                        help="The batchsize to use in training")
    parser.add_argument('--embedding-size', type=int, default=400, help='The size of the embedding.')
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="The difference between losses between iterations \
                        to break.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 or FP32 in training.")
    parser.add_argument("--opt-level", dest="opt_level", type=str, default="O1",
                        help="Which optimisation to use for mixed precision training.")
    parser.add_argument("--eval-metric", type=str, dest="eval_metric", default="accuracy", choices=["accuracy", "attention"])

    parser.add_argument('--workers', default=8, type=int,
                        help="Number of workers for training the network")
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--start-epoch', default=0, type=int, dest="start_epoch",
                        help="The epoch to start/resume training at")
    parser.add_argument('--patience', default=3, type=int)


    # Other arguments
    #parser.add_argument("baseline", type=str, default='oest', choices=["oest", "bare"],
    #:W
    # help="Valid baseline net")

    args = parser.parse_args()

    return args
