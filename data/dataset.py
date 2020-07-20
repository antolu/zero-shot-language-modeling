from math import ceil
from typing import List, Dict, Tuple, Union

import torch
from torch.distributions import Categorical
from torch.utils.data import Dataset as _Dataset
from tqdm import tqdm

from . import create_batch, get_sampling_probabilities, SequenceSequencer

import logging
log = logging.getLogger(__name__)


class Dataset(_Dataset):
    """
    A subclass of the pytorch torch.utils.data.Dataset class, intended to be used with the Pytorch DataLoader for
    training and inference.

    This Dataset class allows to pre-sample languages and sequence lengths to use in training, so the it can be
    used with the Pytorch dataloader to load data deterministically. This is referred to as batch config below.

    See Also
    --------
    make_batches
    DataLoader
    """

    def __init__(self, data: Dict[str, torch.Tensor], batchsize: int, make_config: bool = False, reset_on_iter: bool = False,
                 language_probabilities: torch.Tensor = None, bptt: int = 125,
                 oversample: bool = False, eval: bool = False, ):
        """

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            A data set, with kv pairs {language (str): data (torch.Tensor)}. The data should not have been batchified.
        batchsize : int
            The batch size to use. The data tensors will be reformatted to match the batchsize.
        make_config : bool
            Create a new batch configuration on instantiation of the object if set to True.
        reset_on_iter : bool
            Create new batch configuration on each new iteration of the dataset if True.
        language_probabilities : torch.Tensor
            The probability to sample each language. If left empty, each language is sampled proportional to
            that language data size.
        bptt : int
            Mean sequence length to use when sampling a batch. Passed to the batch maker.
        oversample : Oversample data from the same language when the data runs out. Useful if
            :param language_probabilities is weighted to less well represented languages. Passed to the batch maker.
        eval : bool
            If evaluation mode. If set to True, the sequence length will across samples will be constant, and samples will
            run through languages sequentially instead of sampling it randomly. Passed to the batch maker.
        """

        self.data = dict()
        self.batch_config = None

        self.batchsize = batchsize
        self.reset_on_iter = reset_on_iter

        # save vars to pass to config maker
        self.batch_config_vars = {
            'language_probabilities': language_probabilities,
            'bptt': bptt,
            'oversample': oversample,
            'eval': eval,
        }

        # turn each language data tensor into batches
        for language, language_d in data.items():
            self.data[language] = _batchify(language_d, batchsize)

        self.current = 0

        if make_config:
            self.make_batches()

    def make_batches(self):
        """
        Convenience method. Update the internal batch configuration.
        """
        self.batch_config = make_batches(self.data, **self.batch_config_vars)

    def __len__(self) -> int:
        if self.batch_config is not None:
            return len(self.batch_config)
        else:
            return 0

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Get a sample from the dataset. Requires that the internal batch configuration has been pre-computed.
        Parameters
        ----------
        i : int
            Integral index representing the sample to get.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, int, int]
            A tuple containing (The data, labels, sequence length, language)

        """
        if self.batch_config is None:
            raise RuntimeError('You probably forgot to add "make_config=True in init')

        batch = self.batch_config[i]

        source = self.data[batch['language']]

        data, targets = create_batch(source, batch['seq_len'], batch['start_idx'])

        return data, targets, batch['seq_len'], batch['language']

    def __iter__(self):
        """
        Create a new batch configuration when a new iterator is yielded, if :attr reset_on_iter is set to True.

        Returns
        -------
        Dataset
            An iterator to a Dataset object.
        """
        if self.reset_on_iter:
            log.debug('Creating batch configuration...')
            self.make_batches()

        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.batch_config) - 1:
            batch = self.__getitem__(self.current)
            self.current += 1
            return batch
        else:
            raise StopIteration


def make_batches(data: dict, language_probabilities: torch.Tensor = None, bptt: int = 125, oversample: bool = False,
                 eval: bool = False) -> List[Dict]:
    """
    Prepares data for loading by pre-calculating the batches by sampling languages and sequence lengths at one time.

    Parameters
    ----------
    data : dict
        A dictionary formatted as (language: str, data: torch.Tensor) that contains the batchfied data.
    language_probabilities : torch.Tensor
        A categorical distribution in torch.Tensor representation that represents the probability of sampling a language.
        This parameter is useless if eval is set to True.
    bptt : int
        Backprop through time. Controls the mean of the sequence length distribution, or basically the sequence length
        if eval is set to True.
    oversample : bool
        Oversample from the dataset or not. If true, the language probability distribution will not be adjusted when
        a dataset runs out of data. Instead the language data index will be reset, and the next batch from that language
        will be taken from the beginning of that language set.
    eval : bool
        Make an evaluation batch. If True, the sequence length will be kept constant, and the language data will be gone
        through sequentially instead of sampling the language.

    Returns
    -------
    List:
        A list of dictionaries. Each dictionary with keys 'language', 'seq_len', 'start_idx'`, that allows for creation
        of a batch by calling `data, targets = create_batch(**dictionary)`

    """

    languages = list(data.keys())
    language_to_idx = {lang: i for i, lang in enumerate(languages)}

    batches = list()
    tracking = {lang: {'idx': 0, 'exhausted': False} for lang in languages}

    # create categorical distribution to sample languages from, and sequence length sampler
    lang_probs = get_sampling_probabilities(data, 1.0) if language_probabilities is None else language_probabilities.clone()
    lang_sampler = Categorical(lang_probs)
    seq_len_gen = SequenceSequencer(bptt, constant=eval)

    # To keep track on languages for eval mode
    language_iter = iter(languages)
    current_language = next(language_iter)

    # function handles to sample language to use
    def eval_language():
        nonlocal current_language
        if not tracking[current_language]['exhausted']:
            return current_language
        else:
            current_language = next(language_iter)
            return current_language

    def train_language():
        language_idx = lang_sampler.sample()
        return languages[language_idx]

    if eval:
        get_lang = eval_language
    else:
        get_lang = train_language

    # only used with a progress bar, which is currently disabled.
    approx_iters = 0
    for language, language_data in data.items():
        approx_iters += ceil(language_data.size(0) / bptt)

    while True:
        try:
            language = get_lang()
        except StopIteration:
            break
        language_data = data[language]

        seq_len = seq_len_gen.sample()
        seq_len = min(seq_len, len(language_data - 1 - tracking[language]['idx']))

        batches.append({'language': language, 'seq_len': seq_len, 'start_idx': tracking[language]['idx']})

        tracking[language]['idx'] += seq_len

        # If we run out of data for a particular language set
        if tracking[language]['idx'] >= len(language_data) - 1:
            tracking[language]['idx'] = 0
            tracking[language]['exhausted'] = True

            # If we're all out of data
            if all(d['exhausted'] for _, d in tracking.items()):
                break

            # Change probability distribution to ignore the exhausted dataset
            if not oversample and not eval:
                lang_probs[language_to_idx[language]] = 0
                lang_probs /= lang_probs.sum()

                lang_sampler = Categorical(lang_probs)

    return batches


def _batchify(data: torch.Tensor, batchsize: int) -> torch.Tensor:
    """
    Make batches from a long tensor containing the full data sequence, and trim off
    excess data that makes the tensor non-rectangular. The final tensor will be [batchsize, n_batches big].

    Parameters
    ----------
    data: torch.Tensor :
        The data tensor to refactor
    batchsize: int
        The batchsize with which to refactor the data tensor.

    Returns
    -------
    torch.Tensor
        A refactored tensor with size [batchsize, n_batches]
    """
    n_batches = data.size(0) // batchsize
    data = data.narrow(0, 0, n_batches * batchsize)
    data = data.view(batchsize, -1).t().contiguous()
    return data
