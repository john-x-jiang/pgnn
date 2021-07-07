import os
import numpy as np

from data_loader.heart_data import HeartGraphDataset
from torch_geometric.data import DataLoader


class HeartDataLoader(DataLoader):
    def __init__(self, batch_size, data_dir='data/', split='train', shuffle=True,
                 collate_fn=None, num_workers=1, data_name=None, signal_type=None, num_mesh=None, seq_len=None):
        assert split in ['train', 'valid', 'test']

        self.dataset = HeartGraphDataset(data_dir, data_name, signal_type, num_mesh, seq_len, split)

        super().__init__(self.dataset, batch_size, shuffle, drop_last=True, num_workers=num_workers)
