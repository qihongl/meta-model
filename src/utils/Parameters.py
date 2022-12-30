import os

LOG_ROOT = '../log'


class Parameters():

    def __init__(
        self,
        dim_hidden = 16,
        dim_context = 128,
        ctx_wt = .5,
        penalty_new_context = .1,
        stickiness = .1,
        lr = 1e-3,
        seed = 0,
        dim_input = 30,
        dim_output = 30,
        verbose=True,
    ):
        assert dim_hidden > 0 and dim_input > 0 and dim_output > 0
        assert dim_context >= 0
        assert 1 >= ctx_wt >= 0
        assert penalty_new_context > 0
        assert stickiness > 0
        assert lr > 0

        # network params
        self.dim_hidden = dim_hidden
        self.dim_context = dim_context
        self.dim_input = dim_input
        self.dim_output = dim_output
        # symbolic model params
        self.ctx_wt = ctx_wt
        self.penalty_new_context = penalty_new_context
        self.stickiness = stickiness
        # training param
        self.lr = lr
        # miscs
        self.seed = seed
        # sub_dirs
        sub_dirs = f'dH-{dim_hidden}/dC-{dim_context}-wC-{ctx_wt}/pNew-{penalty_new_context}-s-{stickiness}/lr-{lr}/'

        self.log_dir = os.path.join(LOG_ROOT, sub_dirs)
        self.gen_log_dirs(verbose=verbose)

    def gen_log_dirs(self, verbose=False):
        mkdir_ifdne(self.log_dir, verbose)
        return self.log_dir

    def log_dir_exists(self):
        if os.path.exists(self.log_dir):
            return True
        return False


def mkdir_ifdne(dir_name, verbose=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if verbose:
            print(f'Dir exist: {dir_name}')

if __name__ == "__main__":
    dim_hidden = 16
    dim_context = 128
    ctx_wt = .5
    penalty_new_context = .1
    stickiness = .1
    lr = 1e-3

    p = Parameters(
        dim_hidden = dim_hidden,
        dim_context = dim_context,
        ctx_wt = ctx_wt,
        penalty_new_context = penalty_new_context,
        stickiness = stickiness,
        lr = lr,
    )

    print(p.log_dir)
