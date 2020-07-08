import logging

from torch.utils.data import DataLoader as _DataLoader

from .dataset import Dataset

log = logging.getLogger('zerolm')


class DataLoader(_DataLoader):
    """
    Simple subclass of the Pytorch DataLoader to allow for creation of new batches of data (with differing sequence
    lengths for example) before each iteration.
    """

    def __init__(self, *args, **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        # create new batches upon each new iteration of the dataset
        if isinstance(self.dataset, Dataset) and self.dataset.reset_on_iter:
            log.info('Creating new batches for data loader... This might take a minute or two.')
            self.dataset.make_batches()

        return super().__iter__()
