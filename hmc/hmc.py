from typing import Union

import torch
from torch import nn
from tqdm import trange

from criterion import SplitCrossEntropyLoss
from data import DataLoader
from utils import detach


class HMC:
    def __init__(self, use_apex: bool = None, amp=None, device: Union[torch.device, str] = 'cpu', **kwargs):
        self.use_apex = use_apex
        self.amp = amp
        self.device = device

    def sample(self, model: nn.Module, dataloader: DataLoader,
               loss_function: Union[torch.nn.CrossEntropyLoss, SplitCrossEntropyLoss],
               epsilon: float, total_iter: int, epoch_length: int,
               mass: float = 1.0, optimizer: torch.optim.Optimizer = None,
               need_sample: bool = True, **kwargs):
        """
        Perform HMC sampling

        Parameters
        ----------
        model : nn.Module
            The neural network to sample from. A gaussian prior will be used to provide L2 regularisation.
        dataloader: DataLoader
            Dataloader. Should inherit from torch.utils.data.DataLoader.
        loss_function : Union[torch.nn.CrossEntropyLoss, SplitEntropyLoss]
            Criterion. Required to calculate gradients.
        epsilon : float
            The step size
        total_iter : int
            Total number of iterations to perform
        epoch_length : int
            Number of steps to perform for each 'epoch', i.e. the inner loop
        mass : float
            The mass of the particle in the dynamical system, or in otherwise the variance of the momentum covariance matrix
        optimizer : torch.optim.Optimizer
            A PyTorch optimizer used to calculate gradients if apex is enabled (use_apex is set to True)

        Returns
        -------
        dict
            A dictionary with key-value pairs <param_name: str, param: nn.Parameter>

        """

        mdecay = 0.1
        w_decay = 0.00002

        parameters = {n: p.clone().detach() for n, p in model.named_parameters()}
        parameters_with_gradients = {n: p for n, p in model.named_parameters()}

        momentum = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        momentum_sampler = torch.distributions.normal.Normal(torch.tensor(0.).to(self.device),
                                                             torch.tensor(mass).to(self.device))

        data_iter = iter(dataloader)

        pbar = trange(total_iter)

        for t in range(total_iter // epoch_length):
            for n, p in parameters.items():
                momentum[n] = momentum_sampler.sample(p.shape)

            for l in range(1, epoch_length):
                i = t * epoch_length + l

                try:
                    data, targets, seq_len, lang = next(data_iter)
                except StopIteration:  # out of data, create a new dataloader iterator
                    data_iter = iter(dataloader)
                    data, targets, seq_len, lang = next(data_iter)

                data = data.squeeze(0).to(self.device)
                targets = targets.squeeze(0).to(self.device)
                lang = lang.to(self.device)

                hidden = model.init_hidden(batchsize=data.size(-1))

                hidden = detach(hidden)
                optimizer.zero_grad()

                output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, lang, return_h=True)

                if isinstance(loss_function, SplitCrossEntropyLoss):
                    loss = loss_function(model.decoder.weight, model.decoder.bias, output, targets)
                else:
                    loss = loss_function(output, targets)

                if self.use_apex:
                    with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # first update momentum
                for n, p in parameters_with_gradients.items():
                    if p.grad is not None:
                        momentum[n] = (1.0 - mdecay) * momentum[n] + (-epsilon) * (
                                    p.grad.data + w_decay * parameters[n])
                        if need_sample:
                            momentum[n] = momentum[n] + torch.normal(0, 1, parameters[n].shape).reshape(
                                parameters[n].shape)
                        parameters[n] = parameters[n] + momentum[n]

                pbar.set_description('NLL {:5.4f}'.format(loss.data))
                pbar.update(1)

            # end for  epoch
        # end for total_iter

        return parameters


