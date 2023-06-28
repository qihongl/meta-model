import numpy as np
import torch
log_2pi = np.log(2.0 * np.pi)

def compute_loglik(x, variances):
    """
    Assumes a zero-mean mulitivariate normal with a diagonal covariance function
    Parameters:
        x: array, shape (D,)
            observations
        variances: array, shape (D,)
            Diagonal values of the covariance function
    output
    ------
        log-probability: float
    """
    return -0.5 * (log_2pi * np.shape(x)[0] + np.sum(np.log(variances) + (x**2) / variances ))

def map_variance(samples, nu0=10, var0=.06):
    """
    This estimator assumes an scaled inverse-chi squared prior over the
    variance and a Gaussian likelihood. The parameters d and scale
    of the internal function parameterize the posterior of the variance.
    Taken from Bayesian Data Analysis, ch2 (Gelman)
    samples: N length array or NxD array, where N is the number of
             samples and D is the dimensions
    nu0: prior degrees of freedom
    var0: prior scale parameter
    returns: float or D-length array, mode of the posterior
    ## Calculation ##
    the posterior of the variance is thus (Gelman, 2nd edition, page 50):
        p(var | y) ~ Inv-X^2(nu0 + n, (nu0 * var0 + n * v) / (nu0 + n) )
    where n is the sample size and v is the empirical variance.  The
    mode of this posterior simplifies to:
        mode(var|y) = (nu0 * var0 + n * v) / (v0 + n + 2)
    which is just a weighted average of the two modes
    """
    # get n and v from the data
    n = np.shape(samples)[0]
    v = np.var(samples, axis=0)
    mode = (nu0 * var0 + n * v) / (nu0 + n + 2)
    return mode


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')

dim_input = 5
n = 1
X = np.random.normal(0, 1, size= (n, dim_input))
# noise = np.array([.001, .01, .1, 1, 10])
noise = 1
X_hat = X + np.random.normal(0, 1, size= (n, dim_input)) * noise
np.shape(X_hat)
np.shape(X)
# estimate variance at time t using the data from the past time steps
Sigma = map_variance(X-X_hat)


t = -1
Xt = X[-1]
Xt_hat = X_hat[-1]

# compute the LL for the prediction at time t, which require Sigma
LL = compute_loglik(Xt.reshape(-1) - Xt_hat.reshape(-1), Sigma)
np.exp(LL)


# plt.plot(np.log(Sigma))

np.shape(np.stack([np.ones(30), np.ones(30)]))
