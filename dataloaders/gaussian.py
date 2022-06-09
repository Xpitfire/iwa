import torch
import torchvision.transforms.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math


class DomainAdaptationGaussianDataset(Dataset):
    r"""Domain adaptation version of two Gaussian densities dataset object to iterate and collect samples.
    """
    def __init__(self, data):
        self.xs, self.ys, self.xt, self.yt = data

    @property
    def source_domain_name(self):
        return 'gauss_src'

    @property
    def target_domain_name(self):
        return 'gauss_tgt'

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


def liner_separation(pred, w, b):
    return ((pred[:] @ w - b) > 0)[:, 0]


def create_domain_adaptation_data(config):
    """Creates a domain adaptation version of the moon datasets and dataloader"""
    # parameters
    mean_cls = [0, 0]
    cov_cls = [[1.0, 0.0], [0.0, 1.0]]

    n_s_train = 800
    n_t_train = 800

    n_s_eval = 300
    n_t_eval = 300

    line_coef = np.array([[-0.5], [0.5]])
    line_thresh = 0

    # source domain
    c1 = np.random.multivariate_normal(mean_cls, cov_cls, (n_s_train,))
    mask = liner_separation(c1, line_coef, line_thresh)
    c2 = c1[mask]
    c1 = c1[(1 - mask).astype(np.bool8)]
    Xs_train = np.concatenate([c1, c2], axis=0)
    Ys_train = np.concatenate([np.zeros((len(c1),)), np.ones((len(c2),))], axis=0)
    Xs_train, Ys_train = shuffle(Xs_train, Ys_train)

    c1 = np.random.multivariate_normal(mean_cls, cov_cls, (n_s_eval,))
    mask = liner_separation(c1, line_coef, line_thresh)
    c2 = c1[mask]
    c1 = c1[(1 - mask).astype(np.bool8)]
    Xs_eval = np.concatenate([c1, c2], axis=0)
    Ys_eval = np.concatenate([np.zeros((len(c1),)), np.ones((len(c2),))], axis=0)

    # target domain
    theta = 0.0
    R = np.array([[math.cos(theta), -math.sin(theta)], 
                  [math.sin(theta), math.cos(theta)]])
    t = np.array([0., 0.])


    c1 = np.random.multivariate_normal(mean_cls, cov_cls, (n_t_train,))
    mask = liner_separation(c1, line_coef, line_thresh)
    c2 = c1[mask]
    c1 = c1[(1 - mask).astype(np.bool8)]
    Xt_train = np.concatenate([c1, c2], axis=0)
    Yt_train = np.concatenate([np.zeros((len(c1),)), np.ones((len(c2),))], axis=0)
    Xt_train, Yt_train = shuffle(Xt_train, Yt_train)

    c1 = np.random.multivariate_normal(mean_cls, cov_cls, (n_t_eval,))
    mask = liner_separation(c1, line_coef, line_thresh)
    c2 = c1[mask]
    c1 = c1[(1 - mask).astype(np.bool8)]
    Xt_eval = np.concatenate([c1, c2], axis=0)
    Yt_eval = np.concatenate([np.zeros((len(c1),)), np.ones((len(c2),))], axis=0)

    # rotate and translate 
    Xt_train = (np.dot(R, Xt_train.T).T + t)
    Xt_eval = (np.dot(R, Xt_eval.T).T + t)

    # normalize 
    x_max = np.max(np.concatenate([Xs_train, Xs_eval, Xt_train, Xt_eval], axis=0))
    x_min = np.min(np.concatenate([Xs_train, Xs_eval, Xt_train, Xt_eval], axis=0))

    Xs_train = (Xs_train - x_min) / (x_max - x_min)
    Xt_train = (Xt_train - x_min) / (x_max - x_min)
    Xs_eval = (Xs_eval - x_min) / (x_max - x_min)
    Xt_eval = (Xt_eval - x_min) / (x_max - x_min)

    s_neg_x = Xs_train[Ys_train == 0]
    s_pos_x = Xs_train[Ys_train == 1]
    t_neg_x = Xt_train[Yt_train == 0]
    t_pos_x = Xt_train[Yt_train == 1]

    plt.scatter(s_neg_x[:, 0], s_neg_x[:, 1], c='r', marker='x')
    plt.scatter(s_pos_x[:, 0], s_pos_x[:, 1], c='b', marker='x')
    plt.scatter(t_neg_x[:, 0], t_neg_x[:, 1], c='r', alpha=0.3)
    plt.scatter(t_pos_x[:, 0], t_pos_x[:, 1], c='b', alpha=0.3)
    plt.savefig('tmp/gauss.png')

    train_loaders = []
    eval_loaders = []

    train_loader = DataLoader(
        DomainAdaptationGaussianDataset((Xs_train, Ys_train, Xt_train, Yt_train)),
        batch_size=config.trainer.batchsize,
        shuffle=True
    )
    eval_loader = DataLoader(
        DomainAdaptationGaussianDataset((Xs_eval, Ys_eval, Xt_eval, Yt_eval)),
        batch_size=config.trainer.batchsize,
        shuffle=False
    )


    train_loaders.append(train_loader)
    eval_loaders.append(eval_loader)

    for tr, ev in zip(train_loaders, eval_loaders):
        yield tr, ev
