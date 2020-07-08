import torch
from typing import Tuple
import numpy as np


class SequenceSequencer:
    """
    A class to sequence string sequence lengths from.
    """
    def __init__(self, bptt: int, constant=False):
        """
        Parameters
        ----------
        bptt : int
            The constant sequence length returned by sample if :attr constant is set to True, otherwise serves as
            median for a normal distribution to sample sequence lengths from.
        constant : bool
            If set to true, the sample method will return :attr bptt on each call.
        """
        self.bptt = bptt
        self.constant = constant

    def sample(self) -> int:
        """
        Samples a sequence length.

        Returns
        -------
        int :
            A sequence length, bounded between 5 and 200 if :attr constant is set to False, else :attr bptt
        """
        if self.constant:
            return self.bptt

        bptt = self.bptt if torch.rand(1) < 0.95 else self.bptt / 2

        # Clip of to small or too big sequences
        seq_len = np.clip(
            np.random.normal(loc=bptt, scale=5),
            5, 200)

        return int(seq_len)


def create_batch(source: torch.Tensor, seq_len: int, current_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        The minibatch for training or inference, and the target labels.
    """

    # If we're out of data, take only what we can
    seq_len = min(seq_len, len(source) - 1 - current_idx)

    data = source[current_idx:current_idx + seq_len]
    target = source[current_idx + 1:current_idx + 1 + seq_len].view(-1)

    return data, target


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
