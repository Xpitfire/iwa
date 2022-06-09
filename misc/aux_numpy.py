import numpy as np


def flatten(batch_list):
    return np.concatenate(batch_list, axis=0)


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x[:] - np.max(x, axis=dim)[:, None])
    return e_x / e_x.sum(axis=dim)[:, None]


def predict(probs, dim=1):
    return np.argmax(probs, axis=dim)


def onehot(x, n_classes):
    z = np.zeros((x.size, n_classes), dtype=np.float32)
    z[np.arange(x.size), x] = 1
    return z


def _makearray(a):
    new = np.asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap

def _is_empty_2d(arr):
    # check size first for efficiency
    return arr.size == 0 and np.product(arr.shape[-2:]) == 0

def get_pinv_analysis(a, rcond=1e-15, hermitian=False):
    """Returns temporary values of pseudo inverse computation.
    Tuple consisting of singular values, the cutoff threshold for rcond, 
    and an boolean array indicating larger singular values. 
    """
    a, wrap = _makearray(a)
    rcond = np.asarray(rcond)
    if _is_empty_2d(a):
        m, n = a.shape[-2:]
        res = np.empty(a.shape[:-2] + (n, m), dtype=a.dtype)
        raise ValueError("Matrix is empty!")
    a = a.conjugate()
    u, s, vt = np.linalg.svd(a, full_matrices=False, hermitian=hermitian)

    # discard small singular values
    cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff

    return s, cutoff, large

def pinv_with_singular_values(a, num_singular_values=-1, hermitian=False):
    """Modified pseudoinverse.

    Args:
        a (_type_): the matrix to compute the pseudoinverse for.
        num_singular_values (int, optional): The number of singular values to use. Defaults to -1.
            if -1 uses standard pruning with rcond=1e-15.
    """
    a, wrap = _makearray(a)
    rcond = 1e-15
    rcond = np.asarray(rcond)
    if _is_empty_2d(a):
        m, n = a.shape[-2:]
        res = np.empty(a.shape[:-2] + (n, m), dtype=a.dtype)
        raise ValueError("Matrix is empty!")
    a = a.conjugate()
    u, s, vt = np.linalg.svd(a, full_matrices=False, hermitian=hermitian)

    # discard small singular values
    cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
    if num_singular_values == -1: 
        large = s > cutoff
    else:
        large = np.zeros_like(s, dtype=np.bool)
        for i in range(min(num_singular_values, len(s))):
            large[i] = True
    s = np.divide(1, s, where=large, out=s)
    s[~large] = 0


    res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
    return wrap(res) 