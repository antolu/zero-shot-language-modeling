#!/usr/bin/env python3
import warnings
from typing import List

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class WeightDrop(nn.Module):
    def __init__(self, module: nn.Module, dropout: float, weights_names_ls: list):
        """
        Helper class for implementing weight drop with the weight_drop function.

        Parameters
        ----------
        module: nn.Module
            The module to alter
        weights_names_ls: list
            A list of weight names to apply weight drop to
        dropout: float
            The amount of weight drop to apply

        """
        super().__init__()
        self.weights_names_ls = weights_names_ls
        self.module = module
        self.dropout = dropout

        for name_param in weights_names_ls:
            w = getattr(module, name_param)
            self.module._parameters[name_param] = F.dropout(w, p=self.dropout, training=False)
            self.module.register_parameter(f'{name_param}_raw', nn.Parameter(w.data))

    def forward(self, *args):  # the function formerly known as "forward_new"
        for name_param in self.weights_names_ls:
            param = getattr(self.module, f'{name_param}_raw')
            param_with_dropout = nn.functional.dropout(param, p=self.dropout, training=self.training)
            self.module._parameters[name_param] = param_with_dropout

        self.module.flatten_parameters()

        # forward_res = self.module.forward(*args)
        # return forward_res
        return self.module.forward(*args)


# class WeightDrop(nn.Module):
#     """A module that warps another layer in which some weights will be replaced by 0 during training."""
#
#     def __init__(self, module: nn.Module, p: float, layer_names: List[str] = ['weight_hh_l0']):
#         super(WeightDrop, self).__init__()
#         self.module, self.dropout, self.layer_names = module, p, layer_names
#         self.idxs = [] if hasattr(self.module, '_flat_weights_names') else None
#         for layer in self.layer_names:
#             # Makes a copy of the weights of the selected layers.
#             w = getattr(self.module, layer)
#             self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
#             self.module._parameters[layer] = F.dropout(w, p=self.dropout, training=False)
#             if self.idxs is not None: self.idxs.append(self.module._flat_weights_names.index(layer))
#         if isinstance(self.module, (nn.RNNBase, nn.modules.rnn.RNNBase)):
#             self.module.flatten_parameters = self._do_nothing
#
#     def _setweights(self):
#         "Apply dropout to the raw weights."
#         for i, layer in enumerate(self.layer_names):
#             raw_w = getattr(self, f'{layer}_raw')
#             self.module._parameters[layer] = F.dropout(raw_w, p=self.dropout, training=self.training)
#             if self.idxs is not None: self.module._flat_weights[self.idxs[i]] = self.module._parameters[layer]
#
#     def forward(self, *args):
#         self._setweights()
#         with warnings.catch_warnings():
#             # To avoid the warning that comes because the weights aren't flattened.
#             warnings.simplefilter("ignore")
#             return self.module.forward(*args)
#
#     def reset(self):
#         for layer in self.layer_names:
#             raw_w = getattr(self, f'{layer}_raw')
#             self.module._parameters[layer] = F.dropout(raw_w, p=self.dropout, training=False)
#         if hasattr(self.module, 'reset'): self.module.reset()
#
#     def _do_nothing(self):
#         pass


class LockedDropout(nn.Module):
    """
    Variational dropout. Multiplies with the same mask at each time step with gaussian mask.

    Attributes
    ----------
    dropout: float
        The amount of dropout to apply
    """

    def __init__(self, dropout: float = 0.5):
        """
        Parameters
        ----------
        dropout: float
            The amount of dropout to apply
        """
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.dropout) + ')'


class EmbeddedDropout(nn.Module):
    """
    Applies dropout to the embeddings

    Attributes
    ----------
    dropout: float
        The amount of dropout to apply
    scale: torch.Tensor
        Scale to apply
    """

    def __init__(self, dropout: float = 0.1, scale=None):
        """

        Parameters
        ----------
        dropout : float
            The amount of dropout to apply
            Default is 0.1
        scale : torch.Tensor
            The scale to apply
            Unknown use
        """
        super().__init__()
        self.dropout = dropout
        self.scale = scale

    def __call__(self, embed: nn.Embedding, words):
        if self.dropout and self.training:

            mask = embed.weight.data.clone().resize_((embed.weight.size(0), 1)).bernoulli_(1 - self.dropout).expand_as(
                embed.weight) / (1 - self.dropout)
            masked_embed_weight = mask * embed.weight

        else:
            masked_embed_weight = embed.weight
        if self.scale:
            masked_embed_weight = self.scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type,
                                          embed.scale_grad_by_freq, embed.sparse
                                          )
        return X


if __name__ == '__main__':
    x = torch.randn(2, 1, 10)
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.9)
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    with open('picke_test', 'wb') as f:
        torch.save(wdrnn, f)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('Testing Embedding regularisation')
    print('=-=-=-=-=-=-=-=-=-=')

    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    embed = torch.nn.Embedding(V, h)

    words = torch.randint(low=0, high=V - 1, size=(batch_size, bptt), dtype=torch.long)

    embedded_dropout = EmbeddedDropout()
    origX = embed(words)
    X = embedded_dropout(embed, words)

    print(origX)
    print(X)
