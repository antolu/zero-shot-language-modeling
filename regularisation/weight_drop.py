import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class WeightDrop(Module):
    def __init__(self, module: Module, dropout: float, weights_names_ls: list):
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
            # self.module._parameters[name_param] = F.dropout(w, p=self.dropout, training=False)
            self.module.register_parameter(f'{name_param}_raw', Parameter(w.data))

    def forward(self, *args):  # the function formerly known as "forward_new"
        for name_param in self.weights_names_ls:
            param = getattr(self.module, f'{name_param}_raw')
            param_with_dropout = F.dropout(param, p=self.dropout, training=self.training)
            param_with_dropout = torch.nn.Parameter(param_with_dropout)
            setattr(self.module, name_param, param_with_dropout)

        if isinstance(self.module, torch.nn.RNNBase):
            self.module.flatten_parameters()

        return self.module.forward(*args)


if __name__ == '__main__':
    x = torch.randn(2, 1, 10)
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), dropout=0.9, weights_names_ls=['weight'])
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

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), dropout=0.6, weights_names_ls=['weight_hh_l0'])

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    loss_function = torch.nn.CrossEntropyLoss()

    wdrnn.zero_grad()
    output, hidden = wdrnn(x, h0)
    # targets = torch.randint(10, (2, 10))
    targets = torch.zeros((2, 10), dtype=torch.long)

    loss = loss_function(output, targets)
    loss.backward()

    for n, p in wdrnn.named_parameters():
        print(f'Parameter {n} has grad: {"yes" if p.grad is not None else "no"}')

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

