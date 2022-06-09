import torch
import torchvision.transforms.functional as F
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from hydra.utils import get_original_cwd


class DomainAdaptationMoonDataset(Dataset):
    r"""Domain adaptation version of the moon dataset object to iterate and collect samples.
    """
    def __init__(self, data):
        self.xs, self.ys, self.xt, self.yt = data

    @property
    def source_domain_name(self):
        return 'moons_src'

    @property
    def target_domain_name(self):
        return 'moons_tgt'

    def reset_memory(self):
        pass

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        xs = self.xs[idx]
        ys = self.ys[idx]
        xt = self.xt[idx]
        yt = self.yt[idx]
        # convert to tensors
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(np.array(ys).astype(np.int64))
        xt = torch.from_numpy(xt.astype(np.float32))
        yt = torch.from_numpy(np.array(yt).astype(np.int64))
        return xs, ys, xt, yt


def create_domain_adaptation_data(config):
    """Creates a domain adaptation version of the moon datasets and dataloader"""
    # load data from file

    Xs_train = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_train_x))
    Ys_train = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_train_y)), axis=1)
    Xt_train = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_train_x))
    Yt_train = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_train_y)), axis=1)

    Xs_eval = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_valid_x))
    Ys_eval = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_valid_y)), axis=1)
    Xt_eval = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_valid_x))
    Yt_eval = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_valid_y)), axis=1)

    Xs_test = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_test_x))
    Ys_test = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.source_test_y)), axis=1)
    Xt_test = np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_test_x))
    Yt_test = np.argmax(np.load(os.path.join(get_original_cwd(), config.dataloader.MoonsNS.target_test_y)), axis=1)

    train_loaders = []
    eval_loaders = []
    test_loaders = []

    train_loader = DataLoader(
        DomainAdaptationMoonDataset((Xs_train, Ys_train, Xt_train, Yt_train)),
        batch_size=config.trainer.batchsize,
        shuffle=True
    )
    eval_loader = DataLoader(
        DomainAdaptationMoonDataset((Xs_eval, Ys_eval, Xt_eval, Yt_eval)),
        batch_size=config.trainer.batchsize,
        shuffle=False
    )
    test_loader = DataLoader(
        DomainAdaptationMoonDataset((Xs_test, Ys_test, Xt_test, Yt_test)),
        batch_size=config.trainer.batchsize,
        shuffle=False
    )

    train_loaders.append(train_loader)
    eval_loaders.append(eval_loader)
    test_loaders.append(test_loader)

    for tr, ev, te in zip(train_loaders, eval_loaders, test_loaders):
        yield tr, te
