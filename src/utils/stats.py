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


def get_point_biserial(boundaries_binned, binned_comp, scale=True) -> float:
    '''
    from:
    Bezdek, M., Nguyen, T., Gershman, S. J., Bobick, A., Braver, T. S., & Zacks, J. M. (2022)
    '''
    M_1 = np.mean(binned_comp[boundaries_binned != 0])
    M_0 = np.mean(binned_comp[boundaries_binned == 0])

    n_1 = np.sum(boundaries_binned != 0)
    n_0 = np.sum(boundaries_binned == 0)
    n = n_1 + n_0

    s = np.std(binned_comp)
    r_pb = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n) ** 2))
    if scale:
        num_boundaries = boundaries_binned.astype(bool).sum()
        fake_upper = np.zeros(np.shape(binned_comp), dtype=bool)
        fake_upper[np.argsort(binned_comp)[-num_boundaries:]] = True
        M_1 = np.mean(binned_comp[fake_upper != 0])
        M_0 = np.mean(binned_comp[fake_upper == 0])
        r_upper = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n) ** 2))

        fake_lower = np.zeros(np.shape(binned_comp), dtype=bool)
        fake_lower[np.argsort(binned_comp)[:num_boundaries]] = True
        M_1 = np.mean(binned_comp[fake_lower != 0])
        M_0 = np.mean(binned_comp[fake_lower == 0])
        r_lower = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n) ** 2))
        return (r_pb - r_lower) / (r_upper - r_lower), None
    else:
        return r_pb, None


def padded_corr(event_bound_vec, p_human_bound, shift=True, corr_f=get_point_biserial, porp=.2, step_size=1):
    assert corr_f in [pointbiserialr, pearsonr, get_point_biserial]
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



if __name__ == "__main__":
    x = np.array([0,1,0])
    y = np.array([0.1,.8,.1])
    r = get_point_biserial(x, y, scale=True)

    # z = np.array([-1, -2, -1])
    # print(stable_softmax(z, beta=1/3))
