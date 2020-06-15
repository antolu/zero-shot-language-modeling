from typing import Union

import torch
from torch.nn import CrossEntropyLoss, LSTM

from criterion import SplitCrossEntropyLoss


class DotDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DotDict(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def get_checkpoint(epoch: int, model: LSTM, loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss],
                   optimizer: torch.optim.optimizer, use_apex=False, amp=None, **kwargs):
    """
    Packages network parameters into a picklable dictionary containing keys
    * epoch: current epoch
    * model: the network model
    * loss: the loss function
    * optimizer: the torch optimizer
    * use_apex: use nvidia apex for AMP or not
    * amp: the nvidia AMP object

    Parameters
    ----------
    epoch : int
        The current epoch of training
    model : LSTM
        The network model
    loss_function : SplitCrossEntropyLoss or CrossEntropyLoss
        The loss function
    optimizer : torch.optim.optimizer
        The optimizer function
    use_apex : bool
        If mixed precision mode is activated. If this is true, the `amp` argument should be supplied as well.
        The default value is False.
    amp :
        The nvidia apex amp object, should contain information about state of training
    kwargs :
        Not used

    Returns
    -------
    checkpoint: dict
        A picklable dict containing the checkpoint

    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'loss': loss_function.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if use_apex:
        checkpoint['amp'] = amp.state_dict()

    return checkpoint


def save_model(filepath: str, data):
    """
    Saves a picklable checkpoint to disk
    
    Parameters
    ----------
    filepath : str
        The path to where the checkpoint should be saved.
    data : dict
        A picklable object that is to be saved to disk

    """
    with open(filepath, 'wb') as f:
        torch.save(data, f)


def load_model(filepath: str, model: LSTM, optimizer: torch.optim.optimizer,
               loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss], amp=None, **kwargs):
    """
    Load a checkpointed model into memory by reference
    
    Parameters
    ----------
    filepath : str
        The path to the file on disk containing the checkpoint
    model : LSTM
        The model to load the checkpoint to
    optimizer : torch.optim.optimizer
        The optimizer to load the checkpoint to
    loss_function : SplitCrossEntropyLoss or CrossEntropyLoss
        The loss function to load the checkpoint to
    amp :
        The nvidia apex object to load AMP data to
    kwargs :
        Not used

    Returns
    -------

    """
    with open(filepath, 'rb') as f:
        checkpoint = torch.load(f)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_function.load_state_dict(checkpoint['loss'])

    if amp:
        if 'amp' not in checkpoint:
            raise ValueError('Key amp not in checkpoint. Cannot load apex.')
        amp.load_state_dict(checkpoint['amp'])


def detach(data: Union[torch.Tensor, list]):

    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, tuple) or isinstance(data, list):
        return tuple(detach(d) for d in data)
