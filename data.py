import torch.utils.data as data
from os import path, listdir
from sys import exit
import torch
from tqdm import tqdm
import re

import logging
log = logging.getLogger('log')

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

        # This dict contains all the available datasets in tensor form.
        self.datasets = dict()

        # self.mapping_exists = False

    def load(self):
        """
        Loads the data into memory.
        :return: None
        """

        self.check_dirs()

        mapping_exists = self.mapping_exists()

        if not mapping_exists:
            log.info('Mapping does not exist. Creating it.')
            mapping_filename = path.join(self.datadir, self.character_mappings_name)
        else:
            log.info('Mapping exists. Using cached file. ')
            self.load_mapping()

        self.read_raw_data()

        self.process_data()

        if not mapping_exists:
            self.write_mapping(mapping_filename)

    def load_mapping(self):
        """
        Read the character mapping file from file and saves the mapping in dictionaries.
        :return: The path to the mapping file.
        """
        mapping_file = path.join(self.datadir, self.character_mappings_name)

        with open(mapping_file, 'r') as f:

            for line in f:
                contents = line.split(' ')
                self.character_to_idx[contents[0]] = contents[1]
                self.idx_to_character.append(contents[0])

        log.info('Processed mapping file {}.'.format(mapping_file))

        return mapping_file

    def mapping_exists(self):
        """
        Determines whether a character mapping already exists.
        :return: Boolean whether the mapping exists.
        """
        mapping_file = path.join(self.datadir, self.character_mappings_name)

        return path.exists(mapping_file)

    def check_dirs(self):
        """
        Checks if data directories exists. Will exit with code 1 if they do not.
        :return: None
        """
        if not path.exists(self.datadir):
            log.error('Path {} does not exist'.format(self.datadir))
            exit(1)
        if not path.isdir(self.datadir):
            log.error('{} is not a directory'.format(self.datadir))
            exit(1)

    def read_raw_data(self):
        """
        Reads the raw data into memory and saves the data in string form in dictionaries for further processing
        :return: None
        """
        log.info('Reading in data from file.')
        datadir = path.join(self.datadir, 'bibles_{}'.format(self.dataset))

        languages = listdir(datadir)
        for language in tqdm(languages):
            log.debug('Processing language {}.'.format(language))
            self.data[language] = dict()

            language_data = self.data[language]
            for file in listdir(path.join(datadir, language)):
                key = path.splitext(path.basename(file))[0]
                value = list()
                log.debug('Processing language and split {}/{}.'.format(language, key))

                with open(path.join(datadir, language, file), 'r') as f:
                    for line in f:
                        value.append(line.strip())

                language_data[key] = value

    def process_data(self):
        """
        Processes the raw data in memory into PyTorch tensors, will also create character mapping
        if it does not already exists.
        :return: None
        """
        log.info('Creating data tensors. {}'.format('Creating mapping too.' if self.mapping_exists else ''))

        idx_counter = 0

        regex = re.compile(r'[^\w\s\_\-\?\!\:\,\.]')

        def add_to_tensor(char : str, idx: int, tensor: torch.Tensor):
            """
            Helper function to assign data to PyTorch tensor
            :param char: The character to add to PyTorch tensor
            :param idx: The index of the character in the string
            :param tensor: The tensor which to add the character to
            :return: None
            """
            tensor[idx, 0] = self.character_to_idx[char]

        def add_to_index(char : str, idx: int, tensor: torch.Tensor):
            """
            Helper function to create character mapping. If character does not exist in index, map it and continue to
            create PyTorch tensor.
            :param char: The character to add to the index
            :param idx: The index of the character in the string
            :param tensor: The tensor which to add the character to.
            :return: None
            """
            nonlocal idx_counter

            if char not in self.character_to_idx:
                self.character_to_idx[char] = idx_counter
                idx_counter += 1
            else:
                return

            add_to_tensor(char, idx, tensor)

        # Speeds up computations significantly instead of running if/else statements for each iteration.
        if self.mapping_exists():
            process = add_to_tensor
        else:
            process = add_to_index

        # only for tqdm progress bar
        total = 0
        for _, l in self.data.items():
            total += len(l)

        with tqdm(total=total) as pbar:
            for language, language_d in self.data.items():
                self.datasets[language] = dict()

                for split, split_data in language_d.items():
                    log.debug('Processing language and split {}/{}'.format(language, split))
                    pbar.set_description('Processing language and split {}/{}'.format(language, split))
                    dataset = Dataset(split, language)

                    # Create tensor for each line in data file.
                    for line in split_data:
                        # Remove all characters except for alphanumeric, punctuation and space
                        line = re.sub(regex, '', line)

                        line = line.split(' ')
                        tensor = torch.zeros([len(line), 1])
                        for i in range(len(line)):
                            process(line[i], i, tensor)

                        dataset.data.append(tensor)
                        pbar.update(1)

                    self.datasets[language][split] = dataset

        for ch, idx in self.character_to_idx.items():
            self.idx_to_character.append(ch)

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

        return self.datasets[language][split]


class Dataset(data.Dataset):
    def __init__(self, split: str, language: str):
        """
        Helper class to use with PyTorch data loader
        :param split: Which split this dataset corresponds to: train/validation/test
        :param language: Which language this dataset corresponds to
        """
        self.split = split
        self.language = language
        self.data = list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
