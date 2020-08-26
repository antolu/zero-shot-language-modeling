from torch import nn


class apply_weights:
    def __init__(self, module: nn.Module, parameters: dict):
        """
        Applies the weights in parameters to the module

        Parameters
        ----------
        module : nn.Module
            A Pytorch module
        parameters : dict
            A dictionary containing key-value pairs <param_name: str, param: nn.Parameter>
        """

        self.module = module
        self.parameters = parameters
        self.old_parameters = dict()

    def __enter__(self):
        """
        Set parameters
        """
        state_dict = self.module.state_dict()
        for n, p in self.parameters.items():
            self.old_parameters[n] = state_dict[n]
            state_dict[n] = p

        self.module.load_state_dict(state_dict)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Reset parameters
        """
        state_dict = self.module.state_dict()
        for n, p in self.old_parameters.items():
            state_dict[n] = p

        self.module.load_state_dict(state_dict)
