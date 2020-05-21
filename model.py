import torch.nn as nn

from regularisation import weight_drop, LockedDropout


class LSTM(nn.Module):
    def __init__(self, n_token: int, n_input: int, n_hidden: int, n_layers: int, dropout: float = 0.4,
                 dropouth: float = 0.1, dropouti: float = 0.1, dropoute: float = 0.1,
                 wdrop: float = 0.2, wdrop_layers: list = None):
        """
        Base LSTM for the language model

        Parameters
        ----------
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
        """
        super().__init__()

        self.lstms = [nn.LSTM(n_input if l == 0 else n_hidden, n_hidden if l != n_layers - 1 else n_input, 1, dropout=0)
                      for l in range(n_layers)]

        if wdrop_layers is not None:
            for l, lstm in enumerate(self.lstms):
                if l in wdrop_layers:
                    self.lstms[l] = weight_drop(lstm, ['weight_hh_l0'], wdrop)

        self.lstms = nn.ModuleList(self.lstms)

        self.n_inputs = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.edrop = LockedDropout(dropoute)
        self.idrop = LockedDropout(dropouti)
        self.hdrop = LockedDropout(dropouth)
        self.odrop = LockedDropout(dropout)
        self.embedding_encoder = nn.Embedding(n_token, n_input)

    def forward(self, input, hidden):
        embeddings = self.embedding_encoder(input)

        embeddings = self.edrop(embeddings)

        raw_output = embeddings
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.lstms):
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

        return result, hidden

    def init_hidden(self, batchsize: int) -> []:
        weight = next(self.parameters())

        hidden = [(weight.new_zeros(1, batchsize, self.n_hidden if l != self.n_layers - 1 else self.n_inputs),
                   weight.new_zeros(1, batchsize, self.n_hidden if l != self.n_layers - 1 else self.n_inputs))
                  for l in range(self.n_layers)]

        return hidden
