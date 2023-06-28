from deepsith import iSITH

import scipy.optimize as opt
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', palette='colorblind', context='talk')


def min_fun(x, *args):
    ntau = int(args[2])
    k = int(x[0])
    if k < 4 or k>125:
        return np.inf
    tau_min = args[0]
    tau_max = args[1]
    ev = iSITH(tau_min=tau_min, tau_max=tau_max, buff_max=tau_max*5, k=k, ntau=ntau, dt=1, g=1.0)
    std_0 = ev.filters[:, 0, 0, :].detach().cpu().T.numpy()[::-1].sum(1)[int(tau_min):int(tau_max)].std()
    std_1 = ev.filters[:, 0, 0, :].detach().cpu().T.numpy()[::-1, ::2].sum(1)[int(tau_min):int(tau_max)].std()
    to_min = std_0/std_1
    return to_min

ntau_0 = 16
k_0 = 25
tau_min = 1. #PUT TAU_MIN HERE
tau_max = 200. #PUT TAU_MAX HERE
res = opt.minimize(
    min_fun, [k_0], args=(tau_min, tau_max, ntau_0),
    #method=‘Nelder-Mead’,
    method='Powell',
    #method=‘L-BFGS-B’,
    #method=‘BFGS’,
    #method=‘CG’,
    #method=‘TNC’,
    bounds=[(4, 125)]
)

print(res)
ntau = ntau_0
k = int(res.x)
print('ntau:', ntau, 'k:', k)
ev = iSITH(tau_min=tau_min, tau_max=tau_max, buff_max=tau_max*3, k=k, ntau=ntau, dt=1.0, g=1.0)
f, ax = plt.subplots(1,1, figsize=(8, 4))

# ax.plot(ev.filters[:, 0, 0, :].detach().cpu().T.numpy()[::-1, ::2].sum(1),
#         linewidth=8, color='grey');

# plt.plot(ev.filters[:, 0, 0, :].detach().cpu().T.numpy()[::-1].sum(1),
#         linewidth=8, color='black');
ax.plot(ev.filters[:, 0, 0, :].detach().cpu().T.numpy()[::-1]);
sns.despine()
ev
