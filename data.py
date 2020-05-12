import logging
import re
from glob import glob
from os import path, listdir
from sys import exit

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset as torch_Dataset
from tqdm import tqdm

from utils import DotDict

log = logging.getLogger('log')


class Data:
    def __init__(self, datadir='dataset', dataset='latin'):
        # This dictionary contains all the available datasets in raw string form.
        self.data = dict()
        self.datadir = datadir
        self.dataset = dataset

        self.character_mappings_name = 'character_mappings.txt'
        self.mapping_filename = ''
        self.character_to_idx = dict()
        self.idx_to_character = list()

        self.idx_counter = 0
        self.n_tokens = 0

    def load(self, remap=False, rebuild=False):
        """
        Loads the data into memory.
        :return: None
        """

        check_dir(self.datadir)

        mapping_exists = self.mapping_exists()

        if not mapping_exists or remap:
            mapping_filename = path.join(self.datadir, self.character_mappings_name)
        else:
            self.load_mapping()

        self.process_data(load_data=not rebuild)

        if not mapping_exists or remap:
            self.write_mapping(mapping_filename)

    def process_data(self, load_data=False):
        """
        Process the dataset, one language at a time.
        :return: None
        """

        log.info('{} {}'.format('Creating data tensors.' if not load_data else 'Loading data tensor.',
                                'Creating mapping too.' if self.mapping_exists and not load_data else ''))

        def add_to_tensor(char: str, idx: int, tensor: torch.Tensor):
            """
            Helper function to assign data to PyTorch tensor
            :param char: The character to add to PyTorch tensor
            :param idx: The index of the character in the string
            :param tensor: The tensor which to add the character to
            :return: None
            """
            tensor[idx] = self.character_to_idx[char]

        def add_to_index(char: str, idx: int, tensor: torch.Tensor):
            """
            Helper function to create character mapping. If character does not exist in index, map it and continue to
            create PyTorch tensor.
            :param char: The character to add to the index
            :param idx: The index of the character in the string
            :param tensor: The tensor which to add the character to.
            :return: None
            """

            if char not in self.character_to_idx:
                self.character_to_idx[char] = self.idx_counter
                self.idx_counter += 1

            add_to_tensor(char, idx, tensor)

        # Speeds up computations significantly instead of running if/else statements for each iteration.
        processor = add_to_tensor if self.mapping_exists() else add_to_index

        languages, datadir, pbar = self.__get_tqdm_iterator()
        for language in pbar:
            self.data[language] = dict()

            language_dir = path.join(datadir, language)
            for split in glob(path.join(language_dir, '*.txt')):
                filepath = split
                split = path.splitext(path.basename(filepath))[0]
                savepath = path.splitext(filepath)[0] + '.pt'

                log.debug('Processing language and split {}.'.format(language))
                pbar.set_description('Processing language and split {}'.format(language))

                if not load_data:
                    name, dataset = process_language_data(filepath, processor, language)
                    self.data[language][name] = dataset
                    save_tensor(savepath, dataset)
                else:
                    dataset, split = load_tensor(savepath)
                    self.data[language][dataset.split] = dataset

        for char in self.character_to_idx:
            self.idx_to_character.append(char)

        self.n_tokens = self.idx_counter

    # def get_dataloader(self, split, batchsize, seq_len, language_probabilities):

    def __get_tqdm_iterator(self):
        datadir = path.join(self.datadir, 'bibles_{}'.format(self.dataset))

        languages = listdir(datadir)
        pbar = tqdm(languages)

        return languages, datadir, pbar

    def get_available_languages(self):
        """
        Get which languages exists for the loaded data.
        :return: A list of languages available in the dataset.
        """
        return self.data.keys()

    def get_split(self, split: str, language: str = 'all'):
        """
        Returns the a datqset split based on the argument `split`
        :param split: Valid splits are train, validation, and test
        :param language: The language to get a dataset for
        :return: A torch.utils.data.Dataset object with the data for the specified split.
        """

        if language == 'all':
            output = {lan: splits[split].data if split in splits else None for lan, splits in self.data.items()}

            return output

        return {language: self.data[language][split].data}

    def load_mapping(self):
        """
        Read the character mapping file from file and saves the mapping in dictionaries.
        :return: The path to the mapping file.
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
        """
        Writes the character mapping to file
        :param mapping_file: The path to the character mapping file to write to. This file will be overwritten.
        :return:
        """
        with open(mapping_file, 'w') as f:
            for ch, idx in self.character_to_idx.items():
                print('{} {}'.format(ch, idx), file=f)

    def mapping_exists(self):
        """
        Determines whether a character mapping already exists.
        :return: Boolean whether the mapping exists.
        """
        mapping_file = path.join(self.datadir, self.character_mappings_name)

        return path.exists(mapping_file)

    def __len__(self):
        return len(self.data)


class Dataset(torch_Dataset):
    def __init__(self, split: str, language: str, data: torch.Tensor = None):
        """
        Helper class to use with PyTorch data loader
        :param split: Which split this dataset corresponds to: train/validation/test
        :param language: Which language this dataset corresponds to
        """
        self.split = split
        self.language = language
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class Constant:
    def __init__(self, constant: int):
        self.constant = constant

    def sample(self):
        return self.constant


class DataLoader:
    def __init__(self, data: dict, batchsize: int, seq_len=125, lang_sampler=None, idx_to_language=None,
                 device: str = 'cpu', oversample: bool = True, eval: bool = True):
        self.eval = eval
        self.data = data
        self.seq_len = Constant(seq_len) if isinstance(seq_len, int) else seq_len
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
        self.seq_tracking = DotDict({lan: {'idx': 0, 'exhausted': False} for lan in self.data.keys()})

    def get_batch(self):

        # If in evaluation mode, do not sample language, but run through everything sequentially`
        if self.eval:
            if not self.seq_tracking[self.current_language]['exhausted']:
                language = self.current_language
            else:
                self.current_language = next(self.language_iter)
                language = self.current_language
        else:
            language = self.__get_language()

        seq_len = min(200, int(self.seq_len.sample()))
        data, target = get_batch(self.data[language], seq_len, self.seq_tracking[language], device=self.device)

        if self.eval:
            return data, target
        else:
            return data, target, seq_len

    def get_total_iters(self, seq_len: int):
        if not self.total_iters:
            self.total_iters = 0
            for language, data in self.data.items():
                self.total_iters += self.batchsize * round(data.size(1) / seq_len)

        return self.total_iters

    def is_exhausted(self):
        return all([d['exhausted'] for _, d in self.seq_tracking.items()])

    def __len__(self):
        # TODO: This needs to be implemented
        return 1


def read_raw_data(filepath, regex=re.compile(r'[^\w\s\_\-\?\!\:\,\.]')) -> (str, list, int):
    """
    Reads the file
    :param filepath: The path to the file to be read in
    :param regex: The regex of which characters to keep in the processed data
    :return: Basename of the file (usually the split name), the data in nested lists, number of tokens of this file.
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


def process_language_data(filepath, processor, language) -> (str, Dataset):
    """
    Process a file, wraps read_raw_data and returns the split name and actual dataset.
    :param filepath: Path to the file to be read.
    :param processor: A function handle to process each character (add_to_index or add_to_tensor)
    :param language: The language that is processed
    :return: Split name [str], dataset [Dataset]
    """

    split, split_data, n_tokens = read_raw_data(filepath)

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
    """
    Checks if data directories exists. Will exit with code 1 if they do not.
    :return: None
    """
    if not path.exists(dir):
        log.error('Path {} does not exist'.format(dir))
        exit(1)
    if not path.isdir(dir):
        log.error('{} is not a directory'.format(dir))
        exit(1)


def load_tensor(filepath: str):
    split = path.splitext(path.basename(filepath))[0]
    language = path.split(filepath)[-2]

    tensor = torch.load(filepath)

    dataset = Dataset(split, language, tensor)
    return dataset, split


def save_tensor(filepath: str, data: Dataset):
    torch.save(data.data, filepath)


def batchify(data: torch.Tensor, batchsize: int):
    n_batches = data.size(0) // batchsize
    data = data.narrow(0, 0, n_batches * batchsize)
    data = data.view(batchsize, -1).t().contiguous()
    return data


def get_batch(source: torch.Tensor, seq_len: int, tracking: dict, device='cpu'):
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
    probabilities = torch.zeros(len(datasets))

    i = 0
    probs = torch.tensor([1. * len(data) for _, data in datasets.items()])
    probs /= probs.sum()
    probs = torch.pow(probs, pwr)
    probs /= probs.sum()

    return Categorical(probs)
