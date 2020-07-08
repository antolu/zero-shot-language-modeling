import torch.nn as nn

from typing import Union
from collections import OrderedDict


class MLP(nn.Module):
    """
    A Multi-Layer Perception Block

    Attributes
    ----------
    mlp: torch.nn.sequential
        A torch sequential that is the MLP
    """

    def __init__(self, input_dim: int, dimensions: Union[int, list], activation: str = 'relu', dropout: float = 0.0):
        """

        Parameters
        ----------
        input_dim : int
            The input dimensions to the MLP
        dimensions : int or list
            The intermediate dimensions for the MLP if the argument is a list. If the argument is an integer it represents the size of the output layers, otherwise the size of the output layer is the last element of the list argument.
        activation : str
            The activation method in between linear layers. Only 'relu' is currently supported.
        dropout : float
            The amount of dropout to apply in between each linear layer, before the activation function
        """
        super().__init__()

        assert activation in ['none', 'relu']
        assert 0.0 < dropout < 1.0

        if not isinstance(dimensions, list):
            dimensions = [dimensions]
        dimensions.insert(0, input_dim)

        layers = []
        for lyr in range(len(dimensions) - 1):
            layers.append(('linear' + str(lyr + 1),
                           nn.Linear(dimensions[lyr], dimensions[lyr + 1])))
            if dropout != 0.0 and lyr != len(dimensions) - 2:
                layers.append(('dropout' + str(lyr + 1), nn.Dropout(dropout)))
            if activation.lower() != 'none' and lyr != len(dimensions) - 2:
                layers.append(('relu' + str(lyr + 1), nn.ReLU()))

        self.mlp = nn.Sequential(OrderedDict(layers))

        # Xavier initialisation
        def init_weights(net: nn.Module):
            for m in net.parameters():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal(m.weight)
                    m.bias.fill_(0.01)

        self.mlp.apply(init_weights)

    def forward(self, inputs):
        return self.mlp(inputs)
