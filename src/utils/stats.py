import numpy as np
from scipy.stats import sem, pointbiserialr, pearsonr, spearmanr

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


def padded_corr(
        event_bound_vec, p_human_bound,
        corr_f=get_point_biserial, porp=.1, step_size=1, shift=False,
    ):
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


def erwa(data, decay_factor):
    data = data[:,::-1]
    num_features, num_time_steps = data.shape
    weights = np.power(decay_factor, np.arange(num_time_steps))
    weighted_data = data * weights.reshape(1, -1)
    erwa_result = np.sum(weighted_data, axis=1) / np.sum(weights)
    return erwa_result

def correlate_2RSMs(rsm1, rsm2, use_ranked_r=False):
    """Compute the correlation between 2 RSMs (2nd order correlations)
    Parameters
    ----------
    rsm_i: a 2d array in the form of (n_examples x n_examples)
        a representational similarity matrix

    Returns
    -------
    r: float
        linear_correlation(rsm1, rsm2)
    """
    assert np.shape(rsm1) == np.shape(rsm2)
    # only compare the lower triangular parts (w/o diagonal values)
    rsm1_vec_lower = vectorize_lower_trigular_part(rsm1)
    rsm2_vec_lower = vectorize_lower_trigular_part(rsm2)
    # compute R
    if use_ranked_r:
        r_val, p_val = spearmanr(rsm1_vec_lower, rsm2_vec_lower)
    else:
        r_val, p_val = pearsonr(rsm1_vec_lower, rsm2_vec_lower)
    return r_val, p_val


def vectorize_lower_trigular_part(matrix):
    """Exract the lower triangular entries for a matrix
        useful for computing 2nd order similarity for 2 RDMs, where
        diagonal values should be ignored
    Parameters
    ----------
    matrix: a 2d array

    Returns
    -------
    a vector of lower triangular entries
    """
    assert np.shape(matrix)[0] == np.shape(matrix)[1]
    idx_lower = np.tril_indices(np.shape(matrix)[0], -1)
    return matrix[idx_lower]

if __name__ == "__main__":
    # x = np.array([0,1,0])
    # y = np.array([0.1,.8,.1])
    # r = get_point_biserial(x, y, scale=True)

    # z = np.array([-1, -2, -1])
    # print(stable_softmax(z, beta=1/3))

    boundaries_binned = np.array([0, 0, 1, 0, 0, 1, 0])
    binned_comp = np.array([0, 0, 1, 0, 0, 0, 0])
    r = get_point_biserial(boundaries_binned, binned_comp, scale=True)
    print(r)
