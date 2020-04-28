from os import path, listdir
from sys import exit
import re
from glob import glob

import torch
from torch.utils.data import Dataset as torch_Dataset
from tqdm import tqdm

import logging
log = logging.getLogger('log')


class DataLoader:
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

        log.info('{} {}'.format('Creating data tensors.' if not load_data else 'Loading data tensor.', 'Creating mapping too.' if self.mapping_exists and not load_data else ''))

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

        languages, datadir, pbar = self.get_tqdm_iterator()
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

    def make_batches(self, batchsize, device='cpu', languages=None, split=None, fuzzy=True):
        if languages is None:
            log.info("Processing all datasets into batches.")
            languages = self.data.keys()
        else:
            log.info("Processing languages {} into batches.".format(', '.join(languages)))

        for language in languages:
            for split_name in self.data[language]:
                if split is None or split == split_name or (split in split_name and fuzzy):
                    dataset = self.data[language][split_name]
                    dataset.data = batchify(dataset.data, batchsize).to(device)

    def get_tqdm_iterator(self):
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
            output = list()

            for lan, splits in self.data:
                if split in splits:
                    output.append(splits[split])

            return output

        return self.data[language][split]

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


def read_raw_data(filepath, regex=re.compile(r'[^\w\s\_\-\?\!\:\,\.]')):
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


def process_language_data(filepath, processor, language):
    """
    Process a file, wraps read_raw_data and returns the split name and actual dataset.
    :param filepath: Path to the file to be read.
    :param processor: A function handle to process each character (add_to_index or add_to_tensor)
    :param language: The language that is processed
    :return: Split name [str], dataset [Dataset]
    """

    split, split_data, n_tokens = read_raw_data(filepath)

    dataset = Dataset(split, language)
    tensor = torch.zeros(n_tokens)
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
    data = data.view(batchsize, -1).contiguous()
    return data
