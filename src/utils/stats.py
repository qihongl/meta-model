import numpy as np
from scipy.stats import sem, pointbiserialr, pearsonr


def pad_vector_to_same_length(a, b):
    # compute the padding size
    length_diff = np.abs(len(a) - len(b))
    pad = np.zeros(length_diff, )
    # pad the shorter vector
    if len(a) > len(b):
        b = np.concatenate([b, pad])
    elif len(b) > len(a):
        a = np.concatenate([a, pad])
    else:
        pass
    return a, b


def circular_shift(v, porp = .2, step_size=1):
    shift_size = int(len(v) * porp / 2)
    shifts = np.arange(-shift_size, shift_size, step_size)
    return np.array([np.concatenate([v[-shift:], v[:-shift]]) for shift in shifts])


def padded_corr(event_bound_vec, p_human_bound, shift=True, corr_f=pointbiserialr, porp=.2, step_size=1):
    assert corr_f in [pointbiserialr, pearsonr]
    # compute the padding size
    event_bound_vec, p_human_bound = pad_vector_to_same_length(
        event_bound_vec, p_human_bound
    )
    if shift:
        max_r, max_p = 0, 0
        event_bound_vecs = circular_shift(event_bound_vec, porp, step_size)
        for event_bound_vec in event_bound_vecs:
            r, p = corr_f(event_bound_vec, p_human_bound)
            if r > max_r:
                max_r, max_p = r, p
        return max_r, max_p
    return corr_f(event_bound_vec, p_human_bound)


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
