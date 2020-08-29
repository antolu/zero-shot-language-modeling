import logging
from typing import Dict

import torch
import math

from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


def log_results(name: str, results: Dict[torch.Tensor, torch.Tensor], idx2lang: Dict[torch.Tensor, str],
                tb_writer: SummaryWriter = None):
    """
    Utility function to write results to logger and tensorboard

    Parameters
    ----------
    name : str
        The name of the results, eg. 'Final evaluation results'
    results : dict
        A dictionary with kv pairs <lang: torch.Tensor, loss: torch.Tensor> that will be iterated for data.
    idx2lang : dict
        A dictionary that maps a language index (in torch.Tensor form) to a human readable string.
    tb_writer : SummaryWriter
        (optional) A TensorBoard summarywriter to write to TB

    Returns
    -------

    """
    result_str = '| Language {} | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'
    log.info('=' * 89)
    log.info(f' {name} ')
    log.info('=' * 89)

    tb_str = ''
    for lang, avg_l_loss in results.items():
        langstr = idx2lang[lang]
        result = result_str.format(langstr, avg_l_loss, math.exp(avg_l_loss), avg_l_loss / math.log(2))
        log.info(result)
        tb_str += result + '  \n'
    log.info('=' * 89)

    if tb_writer is not None:
        tb_writer.add_text(f'{name}', tb_str)
        tb_writer.flush()
