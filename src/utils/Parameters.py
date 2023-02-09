import os

class Parameters():

    def __init__(
        self,
        dim_hidden = 16,
        dim_context = 128,
        ctx_wt = .5,
        stickiness = 1,
        gen_grad = 5,
        lik_softmax_beta=.33,
        try_reset_h=False,
        use_shortcut=False,
        pe_tracker_size = 256,
        match_tracker_size = 5,
        n_pe_std = 2,
        lr = 1e-3,
        update_freq = 10,
        subj_id = 0,
        dim_input = 30,
        dim_output = 30,
        verbose=True,
        log_root = '../log',
        exp_name = 'testing',
        dont_make_dir=False,
    ):
        assert dim_hidden > 0 and dim_input > 0 and dim_output > 0
        assert dim_context >= 0
        assert 1 >= ctx_wt >= 0
        assert stickiness >= 0
        assert lr > 0

        # network params
        self.dim_hidden = dim_hidden
        self.dim_context = dim_context
        self.dim_input = dim_input
        self.dim_output = dim_output
        # symbolic model params
        self.ctx_wt = ctx_wt
        self.stickiness = stickiness
        self.lik_softmax_beta = lik_softmax_beta
        self.try_reset_h = try_reset_h
        # short cut param
        self.use_shortcut = use_shortcut
        self.gen_grad = gen_grad
        self.pe_tracker_size = pe_tracker_size
        self.match_tracker_size = match_tracker_size
        self.n_pe_std = n_pe_std
        # training param
        self.lr = lr
        self.update_freq = update_freq
        self.subj_id = subj_id
        # sub_dirs
        self.exp_name = exp_name
        self.log_root = log_root
        sub_dirs = f'dH-{dim_hidden}-dC-{dim_context}-wC-{ctx_wt}/s-{stickiness}-try_reset_h-{try_reset_h}-beta-{lik_softmax_beta}/use_shortcut-{use_shortcut}-gen_grad-{gen_grad}/mtsize-{match_tracker_size}-npe-{n_pe_std}-petsize-{pe_tracker_size}/lr-{lr}-update_freq-{update_freq}/subj_id-{subj_id}/'
        self.log_dir = os.path.join(self.log_root, f'{exp_name}-ckpt', sub_dirs)
        self.fig_dir = os.path.join(self.log_root, f'{exp_name}-fig', sub_dirs)
        self.result_dir = os.path.join(self.log_root, f'{exp_name}-result', sub_dirs)
        self.gen_log_dirs(verbose=verbose)

    def gen_log_dirs(self, verbose=False):
        mkdir_ifdne(self.log_dir, verbose)
        mkdir_ifdne(self.fig_dir, verbose)
        mkdir_ifdne(self.result_dir, verbose)


def mkdir_ifdne(dir_name, verbose=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if verbose:
            print(f'Dir created: {dir_name}')
    else:
        if verbose:
            print(f'Dir exist: {dir_name}')

if __name__ == "__main__":
    dim_hidden = 16
    dim_context = 128
    ctx_wt = .5
    stickiness = .1
    lr = 1e-3

    p = Parameters(
        dim_hidden = dim_hidden,
        dim_context = dim_context,
        ctx_wt = ctx_wt,
        stickiness = stickiness,
        lr = lr,
        dont_make_dir=True # testing
    )

    print(p.log_dir)
