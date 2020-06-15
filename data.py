import logging
import re
from glob import glob
from os import path, listdir
from sys import exit
from pprint import pformat
from typing import Tuple, Union, List, KeysView, Dict
from time import sleep
from multiprocessing import cpu_count, Process, Manager

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset as torch_Dataset
from tqdm import tqdm

from utils import DotDict
from parser import get_args

log = logging.getLogger(__name__)


class Data:
    """ """

    def __init__(self, datadir: str = 'dataset', dataset: str = 'latin', rebuild: bool = True):
        # This dictionary contains all the available datasets in raw string form.
        self.data = dict()
        self.datadir = datadir
        self.dataset = dataset
        self.rebuild = rebuild

        self.character_mappings_name = 'character_mappings.txt'
        self.mapping_filename = ''
        self.character_to_idx = dict()
        self.idx_to_character = list()

        self.idx_counter = 0
        self.n_tokens = 0

    def load(self, rebuild: bool = False):
        """Loads the data into memory.

        Parameters
        ----------
        rebuild: bool :
             (Default value = False)

        Returns
        -------

        """

        check_dir(self.datadir)

        mapping_exists = self.__mapping_exists()

        if not mapping_exists:
            mapping_filename = path.join(self.datadir, self.character_mappings_name)
        else:
            mapping_filename = self.load_mapping()

        self.__process_data(rebuild_cache=rebuild)

        if not mapping_exists or rebuild:
            self.write_mapping(mapping_filename)

    def __process_data(self, rebuild_cache: bool = True):
        """Process the dataset, one language at a time.
        :return: None

        Parameters
        ----------
        rebuild_cache: bool :
             (Default value = True)

        Returns
        -------

        """

        log.info('{} {}'.format('Creating data tensors.' if rebuild_cache else 'Loading data tensor.',
                                'Creating mapping too.' if self.__mapping_exists and rebuild_cache else ''))

        def add_to_tensor(char: str, idx: int, tensor: torch.Tensor):
            """Helper function to assign data to PyTorch tensor

            Parameters
            ----------
            char :
                The character to add to PyTorch tensor
            idx :
                The index of the character in the string
            tensor :
                The tensor which to add the character to

            Returns
            -------
            type
                None

            """
            tensor[idx] = self.character_to_idx[char]

        def add_to_index(char: str, idx: int, tensor: torch.Tensor):
            """Helper function to create character mapping. If character does not exist in index, map it and continue to
            create PyTorch tensor.

            Parameters
            ----------
            char :
                The character to add to the index
            idx :
                The index of the character in the string
            tensor :
                The tensor which to add the character to.

            Returns
            -------
            type
                None

            """

            if char not in self.character_to_idx:
                self.character_to_idx[char] = self.idx_counter
                self.idx_counter += 1

            add_to_tensor(char, idx, tensor)

        # Speeds up computations significantly instead of running if/else statements for each iteration.
        processor = add_to_tensor if self.__mapping_exists() else add_to_index

        datadir = path.join(self.datadir, 'bibles_{}'.format(self.dataset))
        languages = listdir(datadir)

        to_process = list()
        for language in languages:
            language_dir = path.join(datadir, language)
            self.data[language] = dict()

            for split_path in glob(path.join(language_dir, '*.txt')):
                split = path.splitext(path.basename(split_path))[0]
                savepath = path.splitext(split_path)[0] + 'pt'

                to_process.append({
                    'language': language,
                    'split': split,
                    'load_path': split_path,
                    'save_path': savepath,
                })

        def process_split(data: dict, return_list):
            log.debug('Preprocessing language and split {}/{}'.format(data['language'], data['split']))

            split, dataset = _process_language_data(data['load_path'], processor, data['language'])
            save_tensor(data['save_path'], dataset)

            return_list.append(dataset)

        if rebuild_cache:
            with Manager() as manager:
                return_list = manager.list()

                log.debug('Creating multiprocessing workers')
                processes = list()
                for item in to_process:
                    processes.append(Process(target=process_split, args=(item, return_list)))

                active_process = set()
                max_active_processes = cpu_count() - 1

                log.info(f'Starting to preprocess language data with {max_active_processes} workers')
                i = 0
                pbar = tqdm(total=len(to_process))
                while i < len(to_process) or len(active_process) > 0:

                    for p in active_process.copy():
                        p.join(0)
                        if not p.is_alive():
                            pbar.update(1)
                            active_process.remove(p)

                    if len(active_process) <= max_active_processes and i < len(to_process):
                        p = processes[i]
                        p.start()
                        active_process.add(p)
                        i += 1

                    sleep(1e-2)

                for dataset in return_list:
                    log.info(dataset.__dict__)
                    self.data[dataset.language][dataset.split] = dataset.data
        else:
            log.info('Loading preprocessed data tensors')
            pbar = tqdm(to_process)
            for item in pbar:
                split, dataset = load_tensor(item['load_path'])
                self.data[dataset.language][split] = dataset.data

        log.info('Creating character to index mapping')
        for char in self.character_to_idx:
            self.idx_to_character.append(char)

        self.n_tokens = self.idx_counter

        log.info('Done.\nData loading complete.')

    def get_available_languages(self) -> KeysView:
        """Get which languages exists for the loaded data.
        :return: A list of languages available in the dataset.

        Returns
        -------
        KeysView:
            An iterable that contains all the available languages in the dataset

        """
        return self.data.keys()

    def get_split(self, split: str, language: Union[str, list] = 'all') -> Dict:
        """Returns the dataset split based on the argument `split`

        Parameters
        ----------
        split : str
            Valid splits are train, validation, and test
        language : str
            The language to get a dataset for
             (Default value = 'all')
        split: str :

        Returns
        -------
        torch.utils.data.Dataset
            A torch.utils.data.Dataset object with the data for the specified split.

        """

        if language == 'all':
            output = {lan: splits[split].data if split in splits else None for lan, splits in self.data.items()}

            return output
        elif isinstance(language, list):
            raise NotImplementedError

        return {language: self.data[language][split].data}

    def load_mapping(self) -> str:
        """Read the character mapping file from file and saves the mapping in dictionaries.
        :return: The path to the mapping file.

        Returns
        -------
        str
            The absolute path to the mapping file

        """
        mapping_file = path.join(self.datadir, self.character_mappings_name)

        with open(mapping_file, 'r') as f:
            for line in f:
                contents = line.split(' ')
                self.character_to_idx[contents[0]] = int(contents[1])
                self.idx_to_character.append(int(contents[1]))

        log.info('Processed mapping file {}.'.format(mapping_file))

        return mapping_file

    def write_mapping(self, mapping_file: str):
        """Writes the character mapping to file

        Parameters
        ----------
        mapping_file : str
            The path to the character mapping file to write to. This file will be overwritten.

        """
        with open(mapping_file, 'w') as f:
            for ch, idx in self.character_to_idx.items():
                print('{} {}'.format(ch, idx), file=f)

    def __mapping_exists(self) -> bool:
        """Determines whether a character mapping already exists.
        :return: Boolean whether the mapping exists.

        Returns
        -------
        bool
            Whether the character mapping file exists or not

        """
        mapping_file = path.join(self.datadir, self.character_mappings_name)

        return path.exists(mapping_file)

    def __len__(self) -> int:
        return len(self.data)


class Dataset(torch_Dataset):
    def __init__(self, split: str, language: str, data: torch.Tensor = None):
        """Helper class to use with PyTorch data loader

        Parameters
        ----------
        split : str
            Which split this dataset corresponds to: train/validation/test
        language : str
            Which language this dataset corresponds to
        data : torch.Tensor
            The data tensor to wrap

        """
        self.split = split
        self.language = language
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> torch.Tensor:
        return self.data[item]


class Constant:
    """ """

    def __init__(self, constant: int):
        self.constant = constant

    def sample(self) -> int:
        """ """
        return self.constant


class SequenceSequencer:
    def __init__(self, bptt: int, constant=False):
        self.bptt = bptt
        self.constant = constant

    def sample(self):
        if self.constant:
            return self.bptt

        bptt = self.bptt if torch.rand(1) < 0.95 else self.bptt / 2
        distr = torch.normal(bptt, 5)
        seq_len = max(5, int(distr.sample()))
        seq_len = min(seq_len, 200)

        return seq_len


class DataLoader:
    """ """

    def __init__(self, data: dict, batchsize: int, bptt=125, lang_sampler=None, idx_to_language=None,
                 device: Union[str, torch.device] = 'cpu', oversample: bool = True, eval: bool = True):
        self.eval = eval
        self.data = data
        self.seq_len = SequenceSequencer(bptt, constant=self.eval)
        self.bptt = bptt
        self.lang_sampler = get_sampling_probabilities(self.data, 1.0) if lang_sampler is None else lang_sampler
        self.batchsize = batchsize
        self.device = device
        self.oversample = oversample
        self.total_iters = None

        if idx_to_language is None:
            self.idx_to_language = list()
            for language in self.data:
                self.idx_to_language.append(language)

        # Used for eval
        self.language_iter = iter(self.idx_to_language)
        self.current_language = next(self.language_iter)

        self.seq_tracking = DotDict({lan: {'idx': 0, 'exhausted': False} for lan in self.data.keys()})

        self.__make_batches()

    def __make_batches(self):
        dataset = dict()
        for language, data in self.data.items():
            dataset[language] = batchify(data, self.batchsize)
        self.data = dataset

    def __get_language(self):
        language_idx = self.lang_sampler.sample()
        language = self.idx_to_language[language_idx]

        return language

    def reset(self):
        """ """
        self.seq_tracking = DotDict({lan: {'idx': d.idx, 'exhausted': False} for lan, d in self.seq_tracking.items()})

    def reset_all(self):
        """ """
        self.seq_tracking = DotDict({lan: {'idx': 0, 'exhausted': False} for lan in self.data.keys()})

    def get_batch(self) -> Union[Tuple[torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, torch.Tensor, str, int]]:
        """"""

        # If in evaluation mode, do not sample language, but run through everything sequentially`
        if self.eval:
            if not self.seq_tracking[self.current_language]['exhausted']:
                language = self.current_language
            else:
                self.current_language = next(self.language_iter)
                language = self.current_language
        else:
            language = self.__get_language()

        # prevent too big or too small seq_lens
        seq_len = self.seq_len.sample()
        data, target = create_batch(self.data[language], seq_len, self.seq_tracking[language], device=self.device)

        # TODO: Prevent oversampling

        if self.eval:
            return data, target, language
        else:
            return data, target, language, seq_len

    def get_total_iters(self, seq_len: int) -> int:
        """
        Computes the approximate number of iterations to go through the entire dataset. Observer that
        this might be a rough estimate if the sequence length and language to use is sampled from a
        nonuniform distribution with replacement. Therefore this number should be regarded as a naive
        approximation intended to give an estimate of the computational time.

        Parameters
        ----------
        seq_len: int
            The average sequence length
            

        Returns
        -------
        int
            Total number of iterations given the sequence length, batch size, and size of dataset

        """
        if not self.total_iters:
            self.total_iters = 0
            for language, data in self.data.items():
                self.total_iters += self.batchsize * round(data.size(1) / seq_len)

        return self.total_iters

    def get_total_tokens(self):
        """ """
        raise NotImplementedError()

    def is_exhausted(self) -> bool:
        """ """
        return all([d['exhausted'] for _, d in self.seq_tracking.items()])

    def __next__(self):
        if self.is_exhausted():
            raise StopIteration

        return self.get_batch()

    def __iter__(self):
        return self

    def __len__(self) -> int:
        # TODO: This needs to be implemented
        return self.get_total_iters(self.bptt)


# end class Data


def _read_raw_data(filepath, regex=re.compile(r'[^\w\s\_\-\?\!\:\,\.]')) -> Tuple[str, list, int]:
    """Reads the file

    Parameters
    ----------
    filepath :
        The path to the file to be read in
    regex :
        The regex of which characters to keep in the processed data (Default value = re.compile(r'[^\w\s\_\-\?\!\:\)
    \.]') :
        

    Returns
    -------
    (str, list, int)
        Basename of the file (usually the split name), the data in nested lists, number of tokens of this file.

    """

    split = path.splitext(path.basename(filepath))[0]
    data = list()

    n_tokens = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            line = re.sub(regex, '', line)
            tokens = line.split(' ') + ['<']  # < is EOS

            n_tokens += len(tokens)
            data.append(tokens)

    return split, data, n_tokens


def _process_language_data(filepath, processor, language) -> Tuple[str, Dataset]:
    """Process a file, wraps read_raw_data and returns the split name and actual dataset.

    Parameters
    ----------
    filepath :
        Path to the file to be read.
    processor :
        A function handle to process each character (add_to_index or add_to_tensor)
    language :
        The language that is processed

    Returns
    -------
    (str, Dataset)
        Split name [str], dataset [Dataset]

    """

    split, split_data, n_tokens = _read_raw_data(filepath)

    dataset = Dataset(split, language)
    tensor = torch.zeros(n_tokens, dtype=torch.long)
    i = 0

    # Create tensor for each line in data file.
    for line in split_data:
        for char in line:
            processor(char, i, tensor)
            i += 1

    dataset.data = tensor

    return split, dataset


def check_dir(dir: str):
    """Checks if data directories exists. Will exit with code 1 if they do not.
    :return: None

    Parameters
    ----------
    dir: str :
        
    """
    if not path.exists(dir):
        log.error('Path {} does not exist'.format(dir))
        exit(1)
    if not path.isdir(dir):
        log.error('{} is not a directory'.format(dir))
        exit(1)


def load_tensor(filepath: str) -> Tuple[str, Dataset]:
    """

    Parameters
    ----------
    filepath: str
        Path to the file to load
        
    Returns
    -------
    (Dataset, str)
        A Dataset object wrapping the tensor, and the split inferred from the filename
    """
    split = path.splitext(path.basename(filepath))[0]
    language = path.split(filepath)[-2]

    tensor = torch.load(filepath)

    dataset = Dataset(split, language, tensor)
    return split, dataset


def save_tensor(filepath: str, data: Dataset):
    """

    Parameters
    ----------
    filepath: str
        The path to where the tensor shall be saved
    data: Dataset
        A dataset object wrapping the tensor
        
    """
    torch.save(data.data, filepath)


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


def create_batch(source: torch.Tensor, seq_len: int, tracking: dict, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Parameters
    ----------
    source: torch.Tensor
        The base tensor containing all the data for the set, to be sliced off to make the minibatch.
    seq_len: int
        The sequence length of the desired minibatch
    tracking: DotDict
        A dictionary with keys idx and exhausted, to track the progress through each
        language data.
    device : str
         (Default value = 'cpu')

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        The minibatch for training or inference, and the target labels.
    """

    # If we're out of data, take only what we can
    seq_len = min(seq_len, len(source) - 1 - tracking.idx)

    data = source[tracking.idx:tracking.idx + seq_len].to(device)
    target = source[tracking.idx + 1:tracking.idx + 1 + seq_len].view(-1).to(device)

    tracking.idx += seq_len
    if tracking.idx >= len(source) - 1:
        tracking.idx = 0
        tracking.exhausted = True

    return data, target


def get_sampling_probabilities(datasets: dict, pwr: float) -> Categorical:
    """
    In order to allow multi-language training, this method provides a categorical distribution that
    allows for sampling a language to use in training, with probability proportional to the amount of
    data present for each language. This distribution can be further tuned using the `pwr` argument,
    intended to allow for underrepresented languages to be oversampled.

    Parameters
    ----------
    datasets: dict
        A dictionary containing a dataset split with format {language: str, dataset: torch.Tensor}
    pwr: float
        The categorical probabilities will be taken to the power to this value, and then renormalized.
        A value in (0, 1) will make smaller probabilities greater, and for values (1, infnt) big
        probabilities will be overrepresented. Values >1 are thus not recommended.

    Raises
    ------
    ValueError
        If the pwr argument is negative.

    Returns
    -------
    torch.distributions.categorical.Categorical
        A categorical distribution for determining which language to use for training.
    """

    if pwr <= 0:
        raise ValueError('Argument pwr is not allowed to be less than zero.')

    probs = torch.tensor([1. * len(data) for _, data in datasets.items()])
    probs /= probs.sum()
    probs = torch.pow(probs, pwr)
    probs /= probs.sum()

    return Categorical(probs)


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    log.addHandler(ch)

    args = get_args()
    log.info(f'Parsed arguments \n{pformat(args.__dict__)}')

    log.info('Loading data')
    data = Data(args.datadir, args.dataset)
    data.load(args.rebuild)

    log.info('Finished parsing data. Exiting.')
