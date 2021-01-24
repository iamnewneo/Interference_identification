import multiprocessing
from torch.utils.data import DataLoader
from interf_ident.data_loader.dataset import InterfIdentDataset


def create_data_loader(X, y, batch_size):
    cpu_count = multiprocessing.cpu_count()
    ds = InterfIdentDataset(X=X, y=y)
    return DataLoader(ds, batch_size=batch_size, num_workers=cpu_count)