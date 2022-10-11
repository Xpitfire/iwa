import numpy as np
import scipy.io
import torch
import os
from torch.utils.data import Dataset, DataLoader
from hydra.utils import get_original_cwd


DOMAIN_DICT = {'books': 0, 'dvd': 1, 'electronics': 2, 'kitchen': 3}


class AmazonReviewsDataset(Dataset):
    r"""Domain adaptation version of the amazon reviews dataset object to iterate and collect samples.
    """
    def __init__(self, data):
        self.domain, self.x, self.y = data

    @property
    def domain_name(self):
        return self.domain

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # convert to tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(np.array(y).astype(np.int64))
        return x, y


def tf_idf(x):
    """term frequency-inverse document frequency"""
    n = len(x) # number of lines
    n_docs_occ = np.sum(x!=0, axis=0) # number of lines in which word appears
    n_docs_occ = np.where(n_docs_occ==0,1,n_docs_occ) # avoid division of zero
    return x*np.log(n/n_docs_occ)


def split_data(x, y, offset, domain_ix, seed, normalize):
    n_tr_samples = 4000
    
    X = x[offset[domain_ix]:offset[domain_ix+1]]
    Y = y[offset[domain_ix]:offset[domain_ix+1]]

    # normalize data
    if normalize:
        X = tf_idf(X)

    # shuffle data
    np.random.seed(seed)

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]

    # train - test split
    X_train, X_test = np.split(X, [n_tr_samples])
    Y_train, Y_test = np.split(Y, [n_tr_samples])

    # correct length if unequal
    min_len = len(Y_test)
    X_test = X_test[:min_len]
    Y_test = Y_test[:min_len]
    return X_train, X_test, Y_train, Y_test


def read_in_mat_amazon_data(config):
    data = scipy.io.loadmat(os.path.join(get_original_cwd(), config.dataloader.AmazonReviews.data_root, config.dataloader.AmazonReviews.filename))

    x = data['xx'][:config.dataloader.AmazonReviews.n_features, :].toarray().T
    y = data['yy']

    y[y==-1]=0
    y = np.squeeze(y, axis=1)

    offset = data['offset'].flatten()
    return x, y, offset


def ar_data_generator(data_path, domain_id, config, hparams):
    x, y, offset = read_in_mat_amazon_data(config)
    domain_ix = DOMAIN_DICT[domain_id]
    seed = hparams["seed"]
    batch_size = hparams["batch_size"]
    
    if "normalize" in config.dataloader.AmazonReviews and config.dataloader.AmazonReviews.normalize is not None:
        normalize = config.dataloader.AmazonReviews.normalize
    else:
        normalize = False
        
    X_train, X_test, Y_train,Y_test = split_data(x, y, offset, domain_ix, seed, normalize)
    train_loader = DataLoader(
        AmazonReviewsDataset((domain_id, X_train, Y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        AmazonReviewsDataset((domain_id, X_test, Y_test)),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader
