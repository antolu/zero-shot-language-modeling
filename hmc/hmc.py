from typing import Union, List, OrderedDict, Tuple, Dict

import torch
from torch import nn
from tqdm import trange
import numpy as np

from criterion import SplitCrossEntropyLoss
from data import DataLoader
from .evaluator import HMCEvaluator
from utils import detach
from laplace.laplaceprior import _diag_fisher as fisher
import logging

log = logging.getLogger(__name__)


class HMC:
    def __init__(self, model: nn.Module, lr: float, m_decay: float, w_decay: float, num_burn: int, temp: float = 1.0,
                 regulariser: str = 'gamma', use_apex: bool = None, amp=None, alpha: float = 1.0, beta: float = 1.0,
                 w_decay_update: str = 'sep', device: Union[torch.device, str] = 'cpu', **kwargs):
        """

        Parameters
        ----------
        parameters : List[nn.Parameter]
            A list of parameters that we are going to sample. This is only to initalise parameters
        lr : float
            Learning rate, or eta as written in eq 15 in Chen et al.
        m_decay : float
            Momentum decay
        w_decay : float
            Weight decay
        num_burn : int
            Number of burn-in rounds, start averaging after num_burn
        temp : float
            Temperature. temp=0 means no noise during sampling (MAP inference)
        regulariser : str
            Which regulariser to use, choose between gamma, gaussian
        use_apex : bool
            Use Nvidia APEX for AMP
        amp :
            An initialised AMP engine
        device : torch.device or str
            The device to create parameters on
        kwargs :
        """
        self.lr = lr
        self.m_decay = m_decay
        self.temp = temp
        self.num_burn = num_burn
        self.regulariser = regulariser
        self.use_apex = use_apex
        self.amp = amp
        self.device = device

        self.alpha = alpha
        self.beta = beta

        self.model = model

        self.parameters = [n for n, _ in model.named_parameters()]
        self.w_decay = w_decay

        if w_decay_update not in ['sep', 'joint']:
            raise ValueError("w_decay_update must be either 'sep' or 'joint'")
        self.w_decay_update = w_decay_update
        self.w_decay = {n: w_decay for n in self.parameters}

        self.num_train = -1

    def sample(self, dataloader: DataLoader, eval_dataloader: DataLoader,
               loss_function: Union[torch.nn.CrossEntropyLoss, SplitCrossEntropyLoss],
               n_samples: int, step_size: int, optimizer: torch.optim.Optimizer = None,
               sample_every: int = 1, **kwargs) -> Tuple[dict, dict]:
        """
        Perform HMC sampling

        Parameters
        ----------
        dataloader: DataLoader
            Dataloader. Should inherit from torch.utils.data.DataLoader.
        loss_function : Union[torch.nn.CrossEntropyLoss, SplitEntropyLoss]
            Criterion. Required to calculate gradients.
        n_samples : int
            Total number of iterations to perform
        step_size : int
            Number of steps to perform for each simulation, i.e. the inner loop
        optimizer : torch.optim.Optimizer
            A PyTorch optimizer used to calculate gradients if apex is enabled (use_apex is set to True)
        sample_every : int
            Sample every [value] time step

        Returns
        -------
        Tuple[dict, dict]
            A PyTorch state dict containing the sampled parameters of the model

        """

        device = self.device
        model = self.model

        momentum = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        data_iter = iter(dataloader)

        self.num_train = len(dataloader.dataset)
        log.debug(f'New batches, number of training iterations: {self.num_train}')

        evaluator = HMCEvaluator(self.num_burn, self.model, eval_dataloader, loss_function, device=self.device)
        pbar = trange(n_samples)
        i = 0

#        fishers_matrices = fisher(self.model, loss_function, dataloader, optimizer, device=device)
        fishers_matrices = None

        for t in range(self.num_burn + n_samples * sample_every):
            for l in range(1, step_size):
                i = t * step_size + l

                try:
                    data, targets, seq_len, lang = next(data_iter)
                except StopIteration:  # out of data, create a new dataloader iterator
                    # Resample regularisation hyperparameters
                    self.update_hyperparameter()

                    data_iter = iter(dataloader)
                    self.num_train = len(dataloader.dataset)
                    log.debug(f'New batches, number of training iterations: {self.num_train}')

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

                # Update parameters with SGHMC
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        momentum[n] *= (1.0 - self.m_decay)
                        momentum[n] += (-self.lr) * (p.grad.data + self.w_decay[n] * p.data)  # second term is regularizer
                        momentum[n] += torch.normal(torch.zeros_like(p), self.get_sigma()).reshape(p.shape)
                        p.data += momentum[n]

                log.debug(f'Iteration {i} | NLL {loss.item():5.4f}')
                pbar.set_description('NLL {:5.4f}'.format(loss.item()))
                pbar.update(1)

            # end for  epoch
            if (t - self.num_burn) % n_samples == sample_every:
                evaluator(t)
                self.model.train()
        # end for total_iter

        return evaluator.average_log_likelihood, evaluator.log_likelihoods

    def update_hyperparameter(self):
        if self.w_decay_update == 'joint':
            self.__update_parameter({n: p for n, p in self.model.named_parameters()})
        elif self.w_decay_update == 'sep':
            for n, p in self.model.named_parameters():
                self.__update_parameter({n: p})

    def get_sigma(self, fishers_matrix=None):
        if self.num_train == -1:
            raise ValueError('num_train has not been set')
        if self.m_decay - 1.0 > -1e-5:
            scale = self.lr / self.num_train
        else:
            scale = self.lr * self.m_decay / self.num_train

        if fishers_matrix is None:
            return torch.sqrt(torch.tensor(2.0 * self.temp * scale)).to(self.device)
        else:
            alpha = 1 - self.m_decay
            beta_hat = self.lr * fishers_matrix / 2
            return torch.sqrt(2 * (alpha - beta_hat) * self.lr / self.num_train).to(self.device)

    def __update_parameter(self, parameters: Dict[str, nn.Parameter]):

        sum_sqr = 0
        sum_cnt = 0

        for n, p in parameters.items():
            sum_sqr += p.pow(2).sum()
            sum_cnt += p.numel()

        alpha = self.alpha + 0.5 * sum_cnt
        beta = self.beta + 0.5 * sum_sqr

        beta = beta.item()

        if self.temp < 1e-6:
            # if we are doing MAP, take the mode. note: normally MAP adjust is not as well as MCMC
            p_lambda = max(alpha - 1.0, 0.0) / beta
        else:
            p_lambda = np.random.gamma(alpha, 1.0 / beta)

        for n, p in parameters.items():
            self.w_decay[n] = p_lambda / self.num_train
            log.debug(f'[{n}] Changed w_decay to {self.w_decay[n]}')

