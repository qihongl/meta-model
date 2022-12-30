import numpy as np
from scipy.stats import sem
from scipy.stats import pointbiserialr


def padded_pointbiserialr(event_bound_vec, p_human_bound):
    # compute the padding size
    length_diff = np.abs(len(event_bound_vec) - len(p_human_bound))
    pad = np.zeros(length_diff, )
    # pad the shorter vector
    if len(event_bound_vec) > len(p_human_bound):
        p_human_bound = np.concatenate([p_human_bound, pad])
    elif len(p_human_bound) > len(event_bound_vec):
        event_bound_vec = np.concatenate([event_bound_vec, pad])
    else:
        pass
    r, p = pointbiserialr(event_bound_vec, p_human_bound)
    return r, p


def compute_stats(matrix, axis=0, n_se=2, omitnan=False):
    """compute mean and errorbar w.r.t to SE
    Parameters
    ----------
    matrix : type
        Description of parameter `matrix`.
    axis : type
        Description of parameter `axis`.
    n_se : type
        Description of parameter `n_se`.
    Returns
    -------
    type
        Description of returned object.
    """
    if omitnan:
        mu_ = np.nanmean(matrix, axis=axis)
        er_ = sem(matrix, nan_policy='omit', axis=axis) * n_se
    else:
        mu_ = np.mean(matrix, axis=axis)
        er_ = sem(matrix, axis=axis) * n_se
    return mu_, er_
