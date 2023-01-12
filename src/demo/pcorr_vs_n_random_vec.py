'''study mean/max pairwise corr as the number of random vector grows'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import compute_stats
from scipy.stats import pearsonr
sns.set(style='white', palette='colorblind', context='talk')

n_sims = 2
more = 0

f, axes = plt.subplots(1,2, figsize=(12, 5), sharey=True, sharex=True)

d_list = [2**x for x in np.arange(4, 11)]
max_d = max(d_list)
cpal = sns.color_palette('viridis', n_colors=len(d_list))

corrs_dict = {d : [] for d in d_list}

for di, d in enumerate(d_list):
    ns = [2**(k+1) for k in np.arange(int(np.log2(max_d)) + more)]

    def compute_pcorr(d, ns):
        X = np.random.normal(size=(max(ns), d))
        max_corr = np.zeros(len(ns))
        mean_corr = np.zeros(len(ns))
        for i, n in enumerate(ns):
            corr_mat = np.corrcoef(X[:n,:])
            corr_mat_no_diag = corr_mat * (1 - np.eye(n))
            # compute stats
            max_corr[i] = np.max(np.abs(corr_mat_no_diag))
            mean_corr[i] = np.mean(np.abs(corr_mat_no_diag))
        return mean_corr, max_corr, corr_mat_no_diag[np.triu_indices(max_d, k = 1)]


    mean_corr = [None] * n_sims
    max_corr = [None] * n_sims
    for i in range(n_sims):
        mean_corr[i], max_corr[i], corr_mat_no_diag = compute_pcorr(d, ns)
        corrs_dict[d].extend(corr_mat_no_diag)

    mean_corr, max_corr = np.array(mean_corr), np.array(max_corr)
    mean_corr_mu, mean_corr_se = compute_stats(mean_corr)
    max_corr_mu, max_corr_se = compute_stats(max_corr)


    xticks = range(len(ns))
    axes[0].errorbar(x=xticks, y=max_corr_mu, yerr=max_corr_se, label=f'{d}', color=cpal[di])
    axes[1].errorbar(x=xticks, y=mean_corr_mu, yerr=mean_corr_se, color=cpal[di])


axes[0].set_xlabel('# vectors')
axes[0].set_ylabel('pairwise correlation')
axes[0].set_title('max')
axes[1].set_xlabel('# vectors')
axes[1].set_title('mean')

for ax in axes:
    ax.set_xticks(xticks)
    ax.set_xticklabels(ns,rotation=30)
    ax.axhline(0, ls = '--', color='grey')
sns.despine()

f.legend(loc=7, title='dim(context)')
f.tight_layout()
f.subplots_adjust(right=0.85)


# plot the distribution directly
f, ax = plt.subplots(1,1, figsize=(8,5))
for di, d in enumerate(d_list):
    sns.kdeplot(corrs_dict[d], color=cpal[di], ax=ax, label=f'{d}')
ax.set_xlabel('pairwise correlation')
ax.legend()
sns.despine()
