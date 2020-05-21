import torch
import torch.nn as nn


def weight_drop(module: nn.Module, weights_names_ls: list, dropout: float):
    """
    Wrapper to fix the weight drop class by awd-lstm as mentioned in
    https://github.com/pytorch/pytorch/issues/32346
    Code borrowed from https://github.com/pytorch/pytorch/issues/32346#issuecomment-624309620

    Parameters
    ----------
    module: nn.Module
        The module to apply weight drop on
    weights_names_ls: list
        A list of weight names to apply weight drop to
    dropout: float
        The amount of dropout to apply

    Returns
    -------
    nn.Module
        The modified nn.Module with weight drop applied (or rather will be applied on each forward
        pass).
    """
    original_module_forward = module.forward
    forward_with_drop = ForwardWithDrop(module, weights_names_ls, dropout, original_module_forward)
    setattr(module, 'forward', forward_with_drop)
    return module


class ForwardWithDrop(object):
    def __init__(self, module: nn.Module, weights_names_ls: list, dropout: float, original_module_forward):
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
        original_module_forward
            The forward function of the original module

        See Also
        --------
        weight_drop: Weight drop
        """
        self.weights_names_ls = weights_names_ls
        self.module = module
        self.dropout = dropout
        self.original_module_forward = original_module_forward

    def __call__(self, *args, **kwargs):  # the function formerly known as "forward_new"
        for name_param in self.weights_names_ls:
            param = self.module._parameters.get(name_param)
            param_with_dropout = nn.Parameter(
                torch.nn.functional.dropout(param, p=self.dropout, training=self.module.training),
                requires_grad=param.requires_grad)
            self.module._parameters.__setitem__(name_param, param_with_dropout)

        return self.original_module_forward(*args, **kwargs)


class LockedDropout(nn.Module):
    """
    Variational dropout. Multiplies with the same mask at each time step with gaussian mask
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
            + 'p=' + str(self.p) + ')'
