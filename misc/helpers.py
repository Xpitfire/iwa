import random
import torch
import numpy as np
import os
import math
import importlib
from .aux_numpy import softmax
import torch.nn.functional as F
import copy


def shift_data(x, y, shift=[0, 0], theta=0):
    """Data shift helper. Performs rotation and translation."""
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    x = np.tensordot(r, x, axes=([0], [1])) + np.array(shift)[:, np.newaxis]
    return x.T, y


def count_parameters(model):
    """Counts the available parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def import_from_file(model_path):
    """Import from file path"""
    import_file = os.path.basename(model_path).split(".")[0]
    import_root = os.path.dirname(model_path)
    imported = importlib.import_module("%s.%s" % (import_root, import_file))
    return imported


def import_from_package_by_name(object_name, package_root):
    """Import from package by name"""
    package = importlib.import_module(package_root)
    obj = getattr(package, object_name)
    return obj


def load_function(module, function):
    """Loads a function or class based on the module path and function argument"""
    module = os.path.splitext(module)[0].replace("/", ".")
    return import_from_package_by_name(function, module)


def map_reduce(items, key, redux=np.mean):
    """Remaps a dict to a list and reduces values"""
    items = [v[key] for v in items]
    return redux(items)


def load_seed_list(file_name='seed_list.txt'):
    """Loads seeds from a file"""
    with open(file_name, 'r') as f:
        seeds = f.read().split('\n')
    seeds = [int(s) for s in seeds if s is not None and s != '']
    return seeds


def set_seed(seed=None):
    """Sets the global seeds for random, numpy and torch"""
    if seed:
        random.seed(seed)
    if seed:
        np.random.seed(seed)
    if seed:
        torch.manual_seed(seed)
    if seed & torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if seed & torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    r"""Early stops the training if validation loss doesn't improve after a given patience.
    Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_epochs (int): Forces training for a minimum ammount of time.
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
    """

    def __init__(self, trainer, patience=7, min_epochs=0, verbose=False, delta=0, trace_func=print):
        self.trainer = trainer
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, epoch, val_loss):
        if self.patience <= 0:
            return

        if math.isnan(val_loss):
            print("WARNING! Received NaN values for the evaluation loss. Stopping Training.")
            self.early_stop = True
            return

        # ignore early stopping if minimum amount of epochs is not reached
        if epoch <= self.min_epochs:
            return

        score = -val_loss
        # check if already initialized, if not initilize
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(self.trainer.model)
            self.trainer.save_checkpoint(epoch, 'best-checkpoint', overwrite=True)
        # check if score has improved compared to previous version, if no update patient counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # set break criteria
            if self.counter >= self.patience:
                self.early_stop = True
        # otherwise save new best model
        else:
            self.best_score = score
            del self.best_model
            self.best_model = copy.deepcopy(self.trainer.model)
            self.trainer.save_checkpoint(epoch, 'best-checkpoint', overwrite=True)
            self.counter = 0


def resume_checkpoint(model, resume_path, device='cpu'):
    """Resumes training from an existing model checkpoint"""
    print("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=torch.device(device))
    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint['state_dict'])


def ned_torch(x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    """
    Normalized eucledian distance in pytorch.

    Cases:
        1. For comparison of two vecs directly make sure vecs are of size [B] e.g. when using nes as a loss function.
            in this case each number is not considered a representation but a number and B is the entire vector to
            compare x1 and x2.
        2. For comparison of two batch of representation of size 1D (e.g. scores) make sure it's of shape [B, 1].
            In this case each number *is* the representation of the example. Thus a collection of reps
            [B, 1] is mapped to a rep of the same size [B, 1]. Note usually D does decrease since reps are not of size 1
            (see case 3)
        3. For the rest specify the dimension. Common use case [B, D] -> [B, 1] for comparing two set of
            activations of size D. In the case when D=1 then we have [B, 1] -> [B, 1]. If you meant x1, x2 [D, 1] to be
            two vectors of size D to be compare feed them with shape [D].

    https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
    https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753
    """
    # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
    if len(x1.size()) == 1:
        # [K] -> [1]
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
    # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
    # note this special case is needed since var over dim=1 is nan (1 value has no variance).
    elif x1.size() == torch.Size([x1.size(0), 1]):
        # [B, 1] -> [B]
        # Squeeze important to be consistent with .var, otherwise tensors of different sizes come out without the user expecting it
        ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps)).squeeze()
    # common case is if input is a batch
    else:
        # e.g. [B, D] -> [B]
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


def nes_torch(x1, x2, dim=1, eps=1e-8):
    return 1 - ned_torch(x1, x2, dim, eps)


def get_weights_numpy(s_pred_activation, n_s=1., n_t=1.):
    # P_x probability for sample to be from domain x
    # n_x number of samples from domain x
    # importance weight = n_s/n_t * P_target/P_source
    probs = softmax(s_pred_activation)
    return probs[:, 1:] / (probs[:, :1] + 1e-10) * n_s / n_t


def get_weights_torch(s_pred_activation, n_s=1., n_t=1.):
    # P_x probability for sample to be from domain x
    # n_x number of samples from domain x
    # importance weight = n_s/n_t * P_target/P_source
    probs = F.softmax(s_pred_activation, dim=-1)
    return probs[:, 1:] / (probs[:, :1] + 1e-10) * n_s / n_t


def get_train_test_split_idxs(rng: np.random.Generator, n_data_samples: int, train_val_split: float):
    idxs = np.arange(n_data_samples)
    rng.shuffle(idxs)
    split = int(n_data_samples * train_val_split)
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]
    return train_idxs, test_idxs

def acc(preds, labels):
    """Computes the accuracy."""
    assert preds.shape[0] == labels.shape[0]
    return (preds == labels).sum() / preds.shape[0]