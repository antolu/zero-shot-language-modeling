from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict

from regularisation import WeightDrop, LockedDropout, EmbeddedDropout


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


class LSTM(nn.Module):
    def __init__(self, cond_type: str, prior: Tensor, n_token: int, n_input: int, n_hidden: int, n_layers: int,
                 dropout: float = 0.4, dropouth: float = 0.1, dropouti: float = 0.1, dropoute: float = 0.1,
                 wdrop: float = 0.2, wdrop_layers: list = None, tie_weights=False):
        """
        Base LSTM for the language model

        Parameters
        ----------
        cond_type: str
            What to condition the model on. Support choices ['platanios', 'oestling', 'sutskever', 'none']. If the cond_type is none, the argument prior will not be used, and may be set to None
        prior: torch.Tensor
            The prior to condition the model on. Its use depends on the parameter passed in cond_type.
        n_token: int
            Total number of different symbols  / characters in the dataset
        n_input: int
            Size of input to LSTM
        n_hidden: int
            Number of units in each hidden layer
        n_layers: int
            Number of LSTM layers
        dropout: float
            Dropout for output layer
        dropouth: float
            Dropout applied in between hidden layers
        dropouti: float
            Dropout applied to embedding input to LSTM
        dropoute:
            Dropout applied to embedding (before dropouti)
        wdrop: float
            Weight drop probability applied to the LSTM layers
        wdrop_layers: iterable
            A list or tuple specifying which layers to apply weight drop to
        tie_weights: bool
            Flag to use weight tying in training.
        """
        super().__init__()

        cond_type = cond_type.lower()
        self.cond_type = cond_type
        self.n_inputs = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # Dropout layers
        self.edrop = LockedDropout(dropoute)
        self.idrop = LockedDropout(dropouti)
        self.hdrop = LockedDropout(dropouth)
        self.odrop = LockedDropout(dropout)
        self.encoder = nn.Embedding(n_token, n_input)
        self.embedding_dropout = EmbeddedDropout(dropoute)

        if cond_type != 'none':
            weight = prior.clone()
            self.prior = nn.Embedding.from_pretrained(weight)
            self.prior.weight.requires_grad = False
        self.nlangvec = n_hidden // 16

        if cond_type == 'sutskever':
            self.prior2hid = MLP(prior.shape[1], n_hidden)
            self.prior2inp = MLP(prior.shape[1], n_input) if tie_weights else None
        elif cond_type == 'oestling':
            self.prior2vec = [MLP(prior.shape[1], self.nlangvec) for l in range(n_layers)]
            self.prior2vec = nn.ModuleList(self.prior2vec)

        self.rnns = [torch.nn.LSTM((n_input if lay == 0 else n_hidden) + (self.nlangvec * (cond_type == "oestling")),
                                   n_hidden if lay != n_layers - 1 else (n_input if tie_weights else n_hidden), 1,
                                   dropout=0) for lay in range(n_layers)]

        if wdrop_layers and wdrop:
            for l, lstm in enumerate(self.rnns):
                if l in wdrop_layers:
                    self.rnns[l] = WeightDrop(lstm, wdrop, ['weight_hh_l0'])

        self.rnns = nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(n_hidden, n_token)

        self.tie_weights = tie_weights
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self, initrange: float = 0.1):
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, lang: str, return_h=False):
        """
        Forward pass

        Parameters
        ----------
        input : torch.Tensor
            The input to the RNN. Should be a LongTensor
        hidden : torch.Tensor
            The hidden layer from the previous pass
        lang : str
            The language whose data we are running a forward pass on
        return_h : bool
            Return intermediate outputs

        Returns
        -------
        result: torch.Tensor
            The output of the LSTM
        hidden: torch.Tensor
            The last hidden layer of the LSTM
        loss_typ: Unknown or None
            Unknown use
        raw_outputs: List[torch.Tensor] or None
            Contains all the outputs from the intermediate rnns
        outputs: List[torch.tensor] or None
            Contains all the outputs from intermediate rnns after applying variational dropout

        """
        embeddings = self.embedding_dropout(self.encoder, input)

        embeddings = self.edrop(embeddings)

        raw_output = embeddings
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            if self.cond_type == 'oestling':
                prior = self.prior(lang)
                prior_emb = self.prior2vec[l].expand(embeddings.size(0), embeddings.size(1), self.nlangvec)
                raw_output = torch.cat((raw_output, prior_emb), dim=2)

            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)

            # variational dropout
            if l != self.n_layers - 1:
                raw_output = self.hdrop(raw_output)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.odrop(raw_output)
        outputs.append(output)

        result = output.view(output.size(0) * output.size(1), output.size(2))

        if return_h:
            return result, hidden, raw_outputs, outputs
        else:
            return result, hidden

    def init_hidden(self, batchsize: int) -> list:
        """
        Initialize a hidden layer and return it. Currently it is returning a tensor of zeros the size of the inner weight parameters
        Parameters
        ----------
        batchsize : int
            The batchsize to initialise a hidden layer for

        Returns
        -------
        hidden: List[torch.Tensor]
             A list of size (n_layers) containing zero-initialised hidden layer tensors of size (1, batchsize, layer_size)

        """
        weight = next(self.parameters())

        hidden = [(weight.new_zeros(1, batchsize, self.n_hidden if l != self.n_layers - 1 else self.n_inputs),
                   weight.new_zeros(1, batchsize, self.n_hidden if l != self.n_layers - 1 else self.n_inputs))
                  for l in range(self.n_layers)]

        return hidden
