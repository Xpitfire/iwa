# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import itertools
from hydra.utils import get_original_cwd


DOMAIN_DICT = {'books': 0, 'dvd': 1, 'electronics': 2, 'kitchen': 3}

class AmazonReviewsDataset(Dataset):
    r"""Domain adaptation version of the amazon reviews dataset object to iterate and collect samples.
    """
    def __init__(self, data):
        self.s_domain, self.t_domain, self.xs, self.ys, self.xt, self.yt = data

    @property
    def source_domain_name(self):
        return self.s_domain

    @property
    def target_domain_name(self):
        return self.t_domain

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

def tf_idf(x):
    """term frequency-inverse document frequency"""
    n = len(x) # number of lines
    n_docs_occ = np.sum(x!=0, axis=0) # number of lines in which word appears
    n_docs_occ = np.where(n_docs_occ==0,1,n_docs_occ) # avoid division of zero
    return x*np.log(n/n_docs_occ)

def split_data(x,y, offset, source_domain_ix, target_domain_ix, seed, normalize):
    n_tr_samples = 4000
    
    Xs = x[offset[source_domain_ix]:offset[source_domain_ix+1]]
    Ys = y[offset[source_domain_ix]:offset[source_domain_ix+1]]
    Xt = x[offset[target_domain_ix]:offset[target_domain_ix+1]]
    Yt = y[offset[target_domain_ix]:offset[target_domain_ix+1]] 

    # normalize data
    if normalize:
        Xs = tf_idf(Xs)
        Xt = tf_idf(Xt)

    # shuffle data
    np.random.seed(seed)

    idx_s = np.arange(len(Xs))
    idx_t = np.arange(len(Xt))
    np.random.shuffle(idx_s)
    np.random.shuffle(idx_t)
    Xs, Ys = Xs[idx_s], Ys[idx_s]
    Xt, Yt = Xt[idx_t], Yt[idx_t]

    # train - test split
    Xs_train, Xs_test = np.split(Xs, [n_tr_samples])
    Ys_train, Ys_test = np.split(Ys, [n_tr_samples])
    Xt_train, Xt_test = np.split(Xt, [n_tr_samples])
    Yt_train, Yt_test = np.split(Yt, [n_tr_samples])

    # correct length if unequal
    min_len = min(len(Ys_test), len(Yt_test))
    Xs_test, Xt_test = Xs_test[:min_len], Xt_test[:min_len]
    Ys_test, Yt_test = Ys_test[:min_len], Yt_test[:min_len]
    return Xs_train, Xs_test, Ys_train, Ys_test, Xt_train, Xt_test, Yt_train, Yt_test

def read_in_mat_amazon_data(config):
    data = scipy.io.loadmat(os.path.join(get_original_cwd(), config.dataloader.AmazonReviews.data_root, config.dataloader.AmazonReviews.filename))

    x = data['xx'][:config.dataloader.AmazonReviews.n_features, :].toarray().T
    y = data['yy']

    #correction of y label classes=[0,1]
    y[y==-1]=0
    y = np.squeeze(y, axis=1)

    offset = data['offset'].flatten()
    return x, y, offset

def create_domain_adaptation_data(config):
    dataset_comb = itertools.permutations(config.dataloader.AmazonReviews.domains,2)
    train_loaders = []
    test_loaders = []

    x, y, offset = read_in_mat_amazon_data(config)
    for source_domain, target_domain in dataset_comb:
        source_domain_ix = DOMAIN_DICT[source_domain]
        target_domain_ix = DOMAIN_DICT[target_domain]
        seed = config.dataloader.AmazonReviews.seed
        if "normalize" in config.dataloader.AmazonReviews and config.dataloader.AmazonReviews.normalize is not None:
            normalize = config.dataloader.AmazonReviews.normalize
        else:
            normalize = False
        Xs_train, Xs_test, Ys_train,Ys_test, Xt_train, Xt_test, Yt_train, Yt_test = split_data(x,y, offset, source_domain_ix, target_domain_ix,seed, normalize)
        train_loader = DataLoader(
            AmazonReviewsDataset((source_domain, target_domain, Xs_train, Ys_train, Xt_train, Yt_train)),
            batch_size=config.trainer.batchsize,
            shuffle=True
        )
        test_loader = DataLoader(
            AmazonReviewsDataset((source_domain, target_domain, Xs_test, Ys_test, Xt_test, Yt_test)),
            batch_size=config.trainer.batchsize,
            shuffle=False
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    for t, e in zip(train_loaders, test_loaders):
        yield t, e

