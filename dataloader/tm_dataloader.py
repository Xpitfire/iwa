import torch
import torchvision.transforms.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from hydra.utils import get_original_cwd


class DomainAdaptationMoonDataset(Dataset):
    r"""Domain adaptation version of the moon dataset object to iterate and collect samples.
    """
    def __init__(self, data):
        self.x, self.y = data

    @property
    def domain_name(self):
        return 'moons'

    def reset_memory(self):
        pass

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # convert to tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(np.array(y).astype(np.int64))
        return x, y


def tm_data_generator(data_path, domain_id, config, hparams):
    """Creates a domain adaptation version of the moon datasets and dataloader"""
    # load data from file
    if domain_id == '0':
        X_train = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_train_x))
        Y_train = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_train_y)), axis=1)
        X_test = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_test_x))
        Y_test = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_test_y)), axis=1)
    else:
        X_train = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_train_x))
        Y_train = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_train_y)), axis=1)
        X_test = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_test_x))
        Y_test = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_test_y)), axis=1)
        
    batch_size = hparams["batch_size"]
    train_loader = DataLoader(
        DomainAdaptationMoonDataset((X_train, Y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        DomainAdaptationMoonDataset((X_test, Y_test)),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
