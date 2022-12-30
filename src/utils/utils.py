import os
import torch
import numpy as np
from scipy.linalg import qr

ID2CHAPTER = dict({
    '1' : 'breakfast',
    '2' : 'exercising',
    '3' : 'cleaning',
    '4' : 'bathroom'
})


def split_video_id(video_id, to_int=False):
    actor_id, chapter_id, run_id = video_id.split('.')
    if to_int:
        return int(actor_id), int(chapter_id), int(run_id)
    return actor_id, chapter_id, run_id


def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)


def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))


def to_np(torch_tensor):
    return torch_tensor.data.numpy()


def to_sqnp(torch_tensor, dtype=np.float):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)


def random_ortho_mat(dim):
    Q, _ = qr(np.random.randn(dim, dim))
    return Q


def stable_softmax(x, beta=1, subtract_max=True):
    assert beta > 0
    if subtract_max:
        x -= max(x)
    # apply temperture
    z = x / beta
    return np.exp(z) / (np.sum(np.exp(z)) + 1e-010)


def context_to_bound_vec(context):
    diff = np.array([0] + list(np.diff(context)))
    diff[diff!=0] = 1
    return diff


def loss_to_bound_vec(loss, percentile=10):
    loss = to_np(torch.stack(loss))
    threshold = np.percentile(loss, percentile)
    return loss < threshold


if __name__ == "__main__":
    '''how to use'''

    context = np.array([1,1,1,2,2,2,1,1,1,1])
    bound_vec = context_to_bound_vec(context)

    print(context)
    print(bound_vec)
