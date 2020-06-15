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
                   optimizer: torch.optim, use_apex=False, amp=None, **kwargs):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'loss': loss_function.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if use_apex:
        checkpoint['amp'] = amp.state_dict()

    return checkpoint


def save_model(filename: str, data):
    with open(filename, 'wb') as f:
        torch.save(data, f)


def load_model(filename: str, model: LSTM, optimizer: torch.optim,
               loss_function: Union[SplitCrossEntropyLoss, CrossEntropyLoss], amp=None, **kwargs):
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_function.load_state_dict(checkpoint['loss'])

    if amp:
        if 'amp' not in checkpoint:
            raise ValueError('Key amp not in checkpoint. Cannot load apex.')
        amp.load_state_dict(checkpoint['amp'])


def detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, tuple) or isinstance(data, list):
        return tuple(detach(d) for d in data)
