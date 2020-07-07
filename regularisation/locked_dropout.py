from torch.nn import Module


class LockedDropout(Module):
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
        return self.__class__.__name__ + '(p=' + str(self.dropout) + ')'
