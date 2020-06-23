import logging
import re
from glob import glob
from os import path, listdir
from sys import exit
from pprint import pformat
from typing import Tuple, Union, List, Dict
from time import sleep
from multiprocessing import cpu_count, Process, Manager

import torch
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from utils import DotDict
from parser import get_args

log = logging.getLogger(__name__)


class SequenceSequencer:
    def __init__(self, bptt: int, constant=False):
        self.bptt = bptt
        self.constant = constant

    def sample(self):
        if self.constant:
            return self.bptt

        bptt = self.bptt if torch.rand(1) < 0.95 else self.bptt / 2
        distr = torch.distributions.normal.Normal(bptt, 5)
        seq_len = max(5, int(distr.sample()))
        seq_len = min(seq_len, 200)

        return seq_len


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

        assert path.exists(self.datadir)

    def load(self, rebuild: bool = False):
        """Loads the data into memory.

        Parameters
        ----------
        rebuild: bool :
             (Default value = False)

        Returns
        -------

        """

        mapping_filename = path.join(self.datadir, self.character_mappings_name)
        if path.exists(mapping_filename):
            mapping_filename = self.__load_mapping()

        self.__create_cache()

        if not path.exists(mapping_filename) or rebuild:
            self.get_split('train', use_cached=False)
            self.get_split('valid', use_cached=False)
            self.get_split('test', use_cached=False)
            self.__write_mapping(mapping_filename)

    def __create_cache(self):
        log.info('Reading file paths...')

        datadir = path.join(self.datadir, 'bibles_{}'.format(self.dataset))
        languages = listdir(datadir)

        data = dict()
        for language in languages:
            language_dir = path.join(datadir, language)
            data[language] = dict()

            for split_path in glob(path.join(language_dir, '*.txt')):
                split = path.splitext(path.basename(split_path))[0]  # The split name is the file basename
                savepath = path.splitext(split_path)[0] + 'pt'  # Save a torch file to the same directory

                data[language][split] = {
                    'language': language,
                    'split': split,
                    'txt_path': split_path,
                    'pth_path': savepath,
                }

        self.data = data

    def get_available_languages(self) -> list:
        """Get which languages exists for the loaded data.
        :return: A list of languages available in the dataset.

        Returns
        -------
        KeysView:
            An iterable that contains all the available languages in the dataset

        """
        return list(self.data.keys())

    def get_split(self, split: str, languages: Union[str, list] = 'all', use_cached=True) -> Dict:
        """Returns the dataset split based on the argument `split`

        Parameters
        ----------
        split : str
            Valid splits are train, validation, and test
        languages : str or list
            The language(s) to get a dataset split for
             (Default value = 'all')
        use_cached: bool
            Use precalculated data tensors (True), or rebuild the relevant tensors.
            (Default value = True)

        Returns
        -------
        output: dict
            A dictionary with languages as keys, and the torch tensors as values

        """

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

        if languages == 'all':
            languages = self.data.keys()
        elif isinstance(languages, str):
            if languages not in self.data:
                raise ValueError(f'Language {languages} does not exist in memory.')
            languages = [languages]
        elif isinstance(languages, list):
            for lang in languages:
                if lang not in self.data:
                    raise ValueError(f'Language {lang} does not exist in memory.')
                if split not in self.data[lang]:
                    raise ValueError(f'Split {split} does not exist for language {lang}')
        else:
            raise ValueError(f'Language {languages} does not exist in memory.')

        # Speeds up computations significantly instead of running if/else statements for each iteration.
        processor = add_to_tensor if use_cached else add_to_index

        def process_split(data: dict, return_list):
            log.debug('Preprocessing language and split {}/{}'.format(data['language'], data['split']))

            split, dataset = _process_language_data(data['load_path'], processor)
            torch.save(data['save_path'], dataset)

            output = data.copy()
            output[data] = dataset

            return_list.append(output)

        # Prep to load dataset into memory
        to_process = list()
        for lang in languages:
            to_process.append(self.data[lang][split])

        log.info(f'Reading data languages {languages} into memory with split {split}')
        # Rebuild data tensors or load cached ones
        if not use_cached:
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

                    # Register a finished thread to allow for a new one to be created
                    for p in active_process.copy():
                        p.join(0)
                        if not p.is_alive():
                            pbar.update(1)
                            active_process.remove(p)

                    # Start a new process if there are available cores
                    if len(active_process) <= max_active_processes and i < len(to_process):
                        p = processes[i]
                        p.start()
                        active_process.add(p)
                        i += 1
                        continue

                    sleep(1e-2)

                return {output['langauge']: output['data'] for output in return_list}
        else:
            log.info('Loading preprocessed data tensors')
            pbar = tqdm(to_process)

            output = dict()
            for item in pbar:
                tensor = torch.load(item['load_path'])
                output[item['split']] = tensor

            return output

    def make_dataset(self, split: str, languages: Union[str, list] = 'all', batchsize: int = 1) -> Dict:
        """
        Constructs datasets based on the passed parameters. All the processed sts will be wrapped into a dictionary
        that can be fed to a data loader for efficient data loading. The data is made into batches in the same process

        Parameters
        ----------
        split : str
            Which split to find data for
        languages : str or list
            Which language(s) to find data for
            (Default value = 1)
        batchsize : int
            How big batches we are using.

        Returns
        -------
        output: dict
            A dictionary containing {language: Dataset}

        """
        data = self.get_split(split, languages)

        output = dict()
        for lang, d in data.items():
            output[lang] = batchify(d, batchsize)

        return output

    def __load_mapping(self):
        """
            Read the character mapping file from file and saves the mapping in dictionaries.
        """
        mapping_file = path.join(self.datadir, self.character_mappings_name)

        with open(mapping_file, 'r') as f:
            for line in f:
                contents = line.split(' ')
                self.character_to_idx[contents[0]] = int(contents[1])
                self.idx_to_character.append(int(contents[1]))

        log.info('Processed mapping file {}.'.format(mapping_file))

    def __write_mapping(self, mapping_file: str):
        """Writes the character mapping to file

        Parameters
        ----------
        mapping_file : str
            The path to the character mapping file to write to. This file will be overwritten.

        """
        with open(mapping_file, 'w') as f:
            for ch, idx in self.character_to_idx.items():
                print('{} {}'.format(ch, idx), file=f)


# end class Data

class DataLoader:
    """
    Helper class to be able to iterate through a dataset. This should later be replaced with the build in PyTorch
    Dataloader.

    This class requires that the user has already constructed a batch configuration by running `make_batches`.

    See Also
    --------
    make_batches
    """

    def __init__(self, data: Dict, batch_config: List[Dict], device: Union[str, torch.device] = 'cpu'):
        """
        Parameters
        ----------
        data : dict
            A dictionary formatted as (language: str, data: torch.Tensor) that contains the batchfied data.
        batch_config : list
            A dictionary returned by `make_batches`.
        device : str or torch.device
        """

        self.data = data
        self.device = device
        self.batch_config = batch_config
        self.batchsize = list(data.items())[0][1].size(0)

        self.current = 0

    def __len__(self) -> int:
        return len(self.batch_config)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        batch = self.batch_config[i]

        source = self.data[batch['language']]

        data, targets = create_batch(source, batch['seq_len'], batch['start_idx'], device=self.device)

        return data, targets, batch['seq_len'], batch['language']

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.batch_config) - 1:
            batch = self.__getitem__(self.current)
            self.current += 1
            return batch
        else:
            raise StopIteration


# end class DataLoader

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


def _process_language_data(filepath, processor) -> torch.Tensor:
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

    split, split_data, n_tokens = _read_raw_data(filepath)

    tensor = torch.zeros(n_tokens, dtype=torch.long)
    i = 0

    # Create tensor for each line in data file.
    for line in split_data:
        for char in line:
            processor(char, i, tensor)
            i += 1

    return tensor


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


def create_batch(source: torch.Tensor, seq_len: int, current_idx: int, device='cpu') -> Tuple[
    torch.Tensor, torch.Tensor]:
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
    seq_len = min(seq_len, len(source) - 1 - current_idx)

    data = source[current_idx:current_idx + seq_len].to(device)
    target = source[current_idx + 1:current_idx + 1 + seq_len].view(-1).to(device)

    return data, target


def make_batches(data: dict, lang_probabilities: torch.Tensor = None, bptt: int = 125, batchsize: int = 1,
                 oversample: bool = False, eval: bool = False) -> List[Dict]:
    """
    Prepares data for loading by pre-calculating the batches by sampling languages and sequence lengths at one time.

    Parameters
    ----------
    data : dict
        A dictionary formatted as (language: str, data: torch.Tensor) that contains the batchfied data.
    lang_probabilities : torch.Tensor
        A categorical distribution in torch.Tensor representation that represents the probability of sampling a language.
        This parameter is useless if eval is set to True.
    bptt : int
        Backprop through time. Controls the mean of the sequence length distribution, or basically the sequence length
        if eval is set to True.
    batchsize : int
        The batch size to use when batchifying the dataset.
    oversample : bool
        Oversample from the dataset or not. If true, the language probability distribution will not be adjusted when
        a dataset runs out of data. Instead the language data index will be reset, and the next batch from that language
        will be taken from the beginning of that language set.
    eval : bool
        Make an evaluation batch. If True, the sequence length will be kept constant, and the language data will be gone
        through sequentially instead of sampling the language.

    Returns
    -------
    dict:
        A list of dictionaries. Each dictionary with keys 'language', 'seq_len', 'start_idx'`, that allows for creation
        of a batch by calling `data, targets = create_batch(**dictionary)`

    """

    languages = [data.keys()]
    language_to_idx = {lang: i for i, lang in enumerate(languages)}

    batches = list()
    tracking = {lang: {'idx': 0, 'exhausted': False} for lang in languages}

    lang_probs = get_sampling_probabilities(data, 1.0) if lang_probabilities is None else lang_probabilities
    lang_sampler = Categorical(lang_probs)
    seq_len_gen = SequenceSequencer(bptt, constant=eval)

    # To keep track on languages for eval mode
    language_iter = iter(languages)
    current_language = next(language_iter)

    # batchfied_data = {lang: batchify(d, batchsize) for lang, d in data.items()}

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

    while True:
        try:
            language = get_lang()
        except StopIteration:
            break
        language_data = data[language]

        seq_len = seq_len_gen.sample()
        seq_len = min(seq_len, len(language_data - 1 - tracking[language]['idx']))

        batches.append({'language': language, 'seq_len': seq_len, 'start_idx': tracking['idx']})

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


def get_sampling_probabilities(datasets: dict, pwr: float) -> torch.Tensor:
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
    torch.Tensor
        A categorical distribution for determining which language to use for training.
    """

    if pwr <= 0:
        raise ValueError('Argument pwr is not allowed to be less than zero.')

    probs = torch.tensor([1. * len(data) for _, data in datasets.items()])
    probs /= probs.sum()
    probs = torch.pow(probs, pwr)
    probs /= probs.sum()

    return probs


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
