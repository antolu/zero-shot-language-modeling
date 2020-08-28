from collections import OrderedDict

from torch import nn


class apply_weights:
    def __init__(self, module: nn.Module, state_dict: OrderedDict):
        """
        Applies the weights in parameters to the module

        Parameters
        ----------
        module : nn.Module
            A Pytorch module
        state_dict : OrderedDict
            A PyTorch state dict containing the parameters to apply
        """

        self.module = module
        self.state_dict = state_dict
        self.original_parameters = module.state_dict()

    def __enter__(self):
        """
        Set parameters
        """
        self.module.load_state_dict(self.state_dict)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Reset parameters
        """
        self.module.load_state_dict(self.original_parameters)
