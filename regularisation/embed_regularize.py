import torch
import torch.nn.functional as F
from torch.nn import Module, Embedding


class EmbeddedDropout(Module):
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

    def __call__(self, embed: Embedding, words):
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

        X = F.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type,
                        embed.scale_grad_by_freq, embed.sparse)
        return X


if __name__ == '__main__':
    print('Testing Embedding regularisation')
    print('=-=-=-=-=-=-=-=-=-=')

    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    embed = Embedding(V, h)

    words = torch.randint(low=0, high=V - 1, size=(batch_size, bptt), dtype=torch.long)

    embedded_dropout = EmbeddedDropout()
    origX = embed(words)
    X = embedded_dropout(embed, words)

    print(origX)
    print(X)
