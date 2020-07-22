#!/usr/bin/env python

import hashlib
import logging
import re
from glob import glob
from multiprocessing import Manager, Process, cpu_count, Lock
from os import path, listdir
from time import sleep
from typing import Union, Dict, List, Tuple, Set

import torch
from tqdm import tqdm

from data.dictionary import Dictionary
from utils.utils import multithread

log = logging.getLogger(__name__)


class Corpus:
    """ """

    def __init__(self, datadir: str = 'dataset', rebuild: bool = True):
        # This dictionary contains all the available datasets in raw string form.
        self.data = dict()
        self.datadir = datadir
        self.rebuild = rebuild

        self.dictionary = Dictionary()

        self.idx_counter = 0
        self.n_tokens = 0

        assert path.exists(self.datadir)

        self.__create_cache()

    def __create_cache(self):
        """
        Caches the available data sources, this assumes that the root data folder only contains folders with language
        names.
        """
        log.info('Reading file paths...')

        languages = listdir(self.datadir)

        data = dict()
        for language in languages:
            if not path.isdir(path.join(self.datadir, language)):
                continue

            language_dir = path.join(self.datadir, language)
            data[language] = dict()

            for split_path in glob(path.join(language_dir, '*.txt')):
                split = path.splitext(path.basename(split_path))[0]  # The split name is the file basename
                savepath = path.splitext(split_path)[0] + '.pt'  # Save a torch file to the same directory

                data[language][split] = {
                    'language': language,
                    'split': split,
                    'txt_path': split_path,
                    'pth_path': savepath,
                }

        self.data = data

    def get_available_languages(self) -> list:
        """Get which languages exists for the loaded data.

        Returns
        -------
        KeysView:
            An iterable that contains all the available languages in the dataset

        """
        return list(self.data.keys())

    def make_datasets(self, splits: Union[List, Tuple, Dict], force_rebuild: bool = False,
                      ignore_missing: bool = False) -> Dict:
        """Returns the dataset split based on the argument `split`

        Parameters
        ----------
        splits: dict
            A list or tuple in format
            [{
                'languages'         : iterable,
                'split'             : str,
                'invert_include'    : bool,
            }.
            {...}
            ]
            Each dictionary contains information about loading a split.
            A lone dictionary can also be supplied, in which case it is interpreted as a single data split.
        force_rebuild: bool
            Force a rebuild of the dataset tensors even if they exist on disk.
            (Default value = False)

        Returns
        -------
        output: dict
            A dictionary with languages as keys, and the torch tensors as values

        """

        # argument checking
        keys = ('languages', 'split')
        if isinstance(splits, dict):
            for key in keys:
                if key not in splits:
                    raise ValueError(f'Required key {key} does not exist in passed argument splits')
            splits = [splits]
        elif isinstance(splits, list) or isinstance(splits, tuple):
            for split in splits:
                for key in keys:
                    if key not in split:
                        raise ValueError(f'Required key {key} does not exist in passed argument splits')
        else:
            raise ValueError(f'Passed argument of type {type(splits)} is not in a supported format.')

        all_languages = set()

        for split in splits:
            if 'invert_include' in split and split['invert_include']:
                not_included_langs = split['languages']
                included_langs = set(self.data.keys())
                for language in not_included_langs:
                    if language in included_langs:
                        included_langs.remove(language)
                included_langs = list(included_langs)
                included_langs.sort()
                split['languages'] = included_langs
                split['ignore_missing'] = True

        # make sure that all the languages exist before attempting to load
        for split in splits:
            for language in split['languages']:
                if language not in self.data:
                    if ignore_missing or ('ignore_missing' in split and split['ignore_missing']):
                        log.warning(
                            f'Split {split["split"]}: language {language} does not exist in memory. Skipping language.')
                        split['languages'].remove(language)
                    else:
                        raise ValueError(f'Language {language} does not exist in memory.')

        # make sure all splits for each requested language exists before attempting to load
        for split in splits:
            for language in split['languages']:
                if split['split'] not in self.data[language]:
                    if ignore_missing or ('ignore_missing' in split and split['ignore_missing']):
                        log.warning(
                            f'Split {split["split"]} does not exist for language {language}. Skipping language.')
                        split['languages'].remove(language)
                    else:
                        raise ValueError(f'Split {split["split"]} does not exist for language {language}')

        # Create unique id for this dataset
        datastring = ''
        for split in splits:
            datastring += split['split'] + ' '.join(split['languages'])
        datastring = 'corpus.{}.data'.format(hashlib.md5(datastring.encode()).hexdigest())
        save_data_path = path.join(self.datadir, datastring)

        if path.exists(save_data_path) and not force_rebuild:
            log.info('Loading cached dataset...')
            return torch.load(path.join(self.datadir, datastring))
        else:
            log.info('Producing dataset...')

        # Index all languages
        for split in splits:
            for language in split['languages']:
                self.dictionary.add_lang(language)

        # start multiprocessing, first tokenize and index
        def read_and_tokenize(data: dict):
            log.debug('Preprocessing language and split {}/{}'.format(data['language'], data['split']))

            dataset, tokens = _read_and_tokenize(data['txt_path'])

            output = data.copy()
            output['data'] = dataset
            output['tokens'] = tokens

            return output

        # Prep to load dataset into memory
        log.info('Loading data into memory and tokenizing...')
        tokenized_data = multithread(read_and_tokenize, [[self.data[language][split['split']]] for split in splits for language in split['languages']])

        # Index all tokens
        tokens = set()
        for d in tokenized_data:
            tokens.update(d['tokens'])

        for tkn in tokens:
            self.dictionary.add_tkn(tkn)

        # Now to actually create the data tensors
        def build_tensors(data: dict):
            built_tensor = _produce_tensor(data['data'], self.dictionary.tkn2idx)

            output = data.copy()
            output['tensor'] = built_tensor

            return output

        log.info('Constructing data tensors from token index...')
        built_tensors = multithread(build_tensors, [[data] for data in tokenized_data])

        # Build output dictionary
        output = {split['split']: {} for split in splits}
        for item in built_tensors:
            output[item['split']][self.dictionary.lang2idx[item['language']]] = item['tensor']

        # also pass the dictionary
        output['dictionary'] = self.dictionary

        # save dataset using the unique ID
        log.info(f'Saving data tensors and dictionary to {save_data_path}')
        with open(save_data_path, 'wb') as f:
            torch.save(output, f)

        return output


def _read_and_tokenize(filepath) -> Tuple[List[List[str]], Set[str]]:
    """Process a file, wraps read_raw_data and returns the split name and actual dataset.

    Parameters
    ----------
    filepath :
        Path to the file to be read.
    processor :
        A function handle to process each character (add_to_index or add_to_tensor)

    Returns
    -------
    tensor: torch.Tensor
        The torch data tensor of type long

    """

    data = list()
    all_tokens = set()

    n_tokens = 0
    with open(filepath, 'r') as f:
        for line in f:
            tokens = _tokenize(line) + ['<eos>']  # < is EOS

            n_tokens += len(tokens)
            all_tokens.update(tokens)
            data.append(tokens)

    return data, all_tokens


def _produce_tensor(data: List[List[str]], tkn2idx: Dict[str, int]) -> torch.Tensor:
    """
    Create a data tensor using nested lists of strings, and a token index string -> int. The tokens should already
    be indexed in the passed dictionary.

    Parameters
    ----------
    data : list
        A List of lists of strings. Each element in the sublist should be a token, and exist in :attr: tkn2idx
    tkn2idx : dict
        A token to index (str -> int) dictionary, used to build the data tensor.

    Returns
    -------
    A single torch.Tensor with dtype long (int64) representing the entire passed string data.
    """
    n_tokens = sum([len(l) for l in data])

    tensor = torch.zeros(n_tokens, dtype=torch.long)
    i = 0

    # Create tensor for each line in data file.
    for line in data:
        for tkn in line:
            tensor[i] = tkn2idx[tkn]
            i += 1

    return tensor


def _tokenize(string: str) -> List[str]:
    """
    Tokenize a string. Remove special characters, extra whitespace etc. Can be customised to be more flexible
    in tokenization.

    Parameters
    ----------
    string : str
        The string to tokenize.

    Returns
    -------
    list:
        A list of tokens. Each element in the list represents a token.
    """
    regex = re.compile(r'[^\w\s\_\-\?\!\:\,\.]')

    string = string.strip()
    string = re.sub(regex, '', string)

    return string.split(' ')


def batchify(data: torch.Tensor, batchsize: int) -> torch.Tensor:
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

# if __name__ == '__main__':
#     log.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     log.addHandler(ch)
#
#     args = get_args()
#     log.info(f'Parsed arguments \n{pformat(args.__dict__)}')
#
#     log.info('Loading data')
#     data = Corpus(args.datadir)
#
#     splits_info = [
#         {
#             'split': 'train',
#             'languages': [],
#             'invert_include': True,
#         },
#         {
#             'split': 'test',
#             'languages': ['eng', 'fra'],
#         }
#     ]
#
#     splits = data.make_datasets(splits_info)
#
#     log.info('Processed splits\n' + ' '.join(splits['train'].keys()) + '\n' + ' '.join(splits['test'].keys()))
#
#     log.info('Finished parsing data. Exiting.')
