'''study mean/max pairwise corr as the number of random vector grows'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import compute_stats
from scipy.stats import pearsonr
sns.set(style='white', palette='colorblind', context='talk')

for d in [2**x for x in np.arange(4, 11)]:

    n_sims = 15
    # d = 64
    more = 0
    ns = [2**(k+1) for k in np.arange(int(np.log2(d)) + more)]

    def compute_pcorr(d, ns):
        X = np.random.normal(size=(max(ns), d))
        max_corr = np.zeros(len(ns))
        mean_corr = np.zeros(len(ns))
        for i, n in enumerate(ns):
            corr_mat = np.corrcoef(X[:n,:])
            corr_mat_no_diag = corr_mat * (1 - np.eye(n))
            # compute stats
            max_corr[i] = np.max(corr_mat_no_diag)
            mean_corr[i] = np.mean(corr_mat_no_diag)
        return mean_corr, max_corr


    mean_corr = [None] * n_sims
    max_corr = [None] * n_sims
    for i in range(n_sims):
        mean_corr[i], max_corr[i] = compute_pcorr(d, ns)
    mean_corr, max_corr = np.array(mean_corr), np.array(max_corr)
    mean_corr_mu, mean_corr_se = compute_stats(mean_corr)
    max_corr_mu, max_corr_se = compute_stats(max_corr)


    xticks = range(len(ns))
    # xticks = ns
    f, ax = plt.subplots(1,1, figsize=(7,4))
    ax.errorbar(x=xticks, y=max_corr_mu, yerr=max_corr_se, label='max')
    ax.errorbar(x=xticks, y=mean_corr_mu, yerr=mean_corr_se, label='mean')

    ax.legend()
    ax.set_xlabel('# vectors')
    ax.set_ylabel('pairwise correlation')
    ax.set_xticks(xticks)
    ax.set_ylim([-.05, 1])
    ax.set_xticklabels(ns,rotation=30)
    ax.set_title(f'dim = {d}\npairwise corr when #vecs = dim = %.2f' % max_corr_mu[int(np.log2(d))-1])
    sns.despine()
