from interf_ident import config
import multiprocessing
from torch.utils.data import DataLoader
from interf_ident.data_loader.dataset import InterfIdentDataset


def create_data_loader(X, y, batch_size, preprocessing=None, transform=None):
    ds = InterfIdentDataset(X=X, y=y, preprocessing=preprocessing, transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=config.N_WORKER)
