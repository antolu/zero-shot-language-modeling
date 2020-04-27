from os import path, listdir
from sys import exit
import re

import torch
import torch.utils.data as data
from tqdm import tqdm

import logging
log = logging.getLogger('log')


def read_raw_data(filepath, regex=re.compile(r'[^\w\s\_\-\?\!\:\,\.]')):
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


class DataLoader():
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

        # self.mapping_exists = False

    def load(self):
        """
        Loads the data into memory.
        :return: None
        """

        check_dir(self.datadir)

        mapping_exists = self.mapping_exists()

        if not mapping_exists:
            log.info('Mapping does not exist. Creating it.')
            mapping_filename = path.join(self.datadir, self.character_mappings_name)
        else:
            log.info('Mapping exists. Using cached file. ')
            self.load_mapping()

        self.process_data()

        if not mapping_exists:
            self.write_mapping(mapping_filename)

    def process_data(self):

        datadir = path.join(self.datadir, 'bibles_{}'.format(self.dataset))

        log.info('Creating data tensors. {}'.format('Creating mapping too.' if self.mapping_exists else ''))

        def add_to_tensor(char : str, idx: int, tensor: torch.Tensor):
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

        languages = listdir(datadir)
        pbar = tqdm(languages)
        for language in pbar:
            log.debug('Processing language and split {}.'.format(language))
            pbar.set_description('Processing language and split {}'.format(language))
            self.data[language] = dict()
            for split in listdir(path.join(datadir, language)):
                filepath = path.join(datadir, language, split)
                name, dataset = process_language_data(filepath, processor, language)

                self.data[language][name] = dataset

        for char in self.character_to_idx:
            self.idx_to_character.append(char)

        self.n_tokens = self.idx_counter

    def write_mapping(self, mapping_file: str):
        """
        Writes the character mapping to file
        :param mapping_file: The path to the character mapping file to write to. This file will be overwritten.
        :return:
        """
        with open(mapping_file, 'w') as f:
            for ch, idx in self.character_to_idx.items():
                print('{} {}'.format(ch, idx), file=f)

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

            for lan, splits in self.datasets:
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

    def mapping_exists(self):
        """
        Determines whether a character mapping already exists.
        :return: Boolean whether the mapping exists.
        """
        mapping_file = path.join(self.datadir, self.character_mappings_name)

        return path.exists(mapping_file)

    def __len__(self):
        return len(self.data)


class Dataset(data.Dataset):
    def __init__(self, split: str, language: str):
        """
        Helper class to use with PyTorch data loader
        :param split: Which split this dataset corresponds to: train/validation/test
        :param language: Which language this dataset corresponds to
        """
        self.split = split
        self.language = language
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
