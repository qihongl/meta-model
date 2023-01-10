import os

class Parameters():

    def __init__(
        self,
        dim_hidden = 16,
        dim_context = 128,
        ctx_wt = .5,
        penalty_new_context=0,
        stickiness = 1,
        lik_softmax_beta=.33,
        lr = 1e-3,
        update_freq = 10,
        subj_id = 0,
        dim_input = 30,
        dim_output = 30,
        verbose=True,
        log_root = '../log',
        dont_make_dir=False,
    ):
        assert dim_hidden > 0 and dim_input > 0 and dim_output > 0
        assert dim_context >= 0
        assert 1 >= ctx_wt >= 0
        assert penalty_new_context >= 0
        assert stickiness >= 0
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
        self.lik_softmax_beta = lik_softmax_beta
        # training param
        self.lr = lr
        self.update_freq = update_freq
        self.subj_id = subj_id
        # sub_dirs
        self.log_root = log_root
        sub_dirs = f'dH-{dim_hidden}/dC-{dim_context}-wC-{ctx_wt}/pNew-{penalty_new_context}-s-{stickiness}-beta-{lik_softmax_beta}/lr-{lr}-update_freq-{update_freq}/subj_id-{subj_id}/'
        self.log_dir = os.path.join(self.log_root, sub_dirs, 'ckpt')
        self.fig_dir = os.path.join(self.log_root, sub_dirs, 'fig')
        self.result_dir = os.path.join(self.log_root, sub_dirs, 'result')
        print(self.log_dir)
        print(self.fig_dir)
        print(self.result_dir)
        self.gen_log_dirs(verbose=verbose)

    def gen_log_dirs(self, verbose=False):
        mkdir_ifdne(self.log_dir, verbose)
        mkdir_ifdne(self.fig_dir, verbose)
        mkdir_ifdne(self.result_dir, verbose)


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
        dont_make_dir=True # testing
    )

    print(p.log_dir)
