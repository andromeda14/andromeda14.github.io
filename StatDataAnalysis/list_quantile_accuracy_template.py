import numpy as np
import pandas as pd
import statsmodels
import scipy.stats
import seaborn as sns

import matplotlib.pyplot as plt


## (a)


n = 50
shape = 3
rate = 0.5

# TODO:
# generate sample
# as always check which parameters you want to set and what kind of
# parameters are in scipy.stats
# see: scipy.stats.gamma.rvs




# TODO:
# plot histogram of a sample and add denisity function of theoretical 
# distribution with given parameters





# TODO:
# find quantiles for theoretical distributio and for the sample

# compare empirical and theoretical quantiles (e.g. on plot)
# and repeat it for few different samples

plt.scatter(quantiles_exact, quantiles_approx)
plt.plot([0, 20], [0, 20])

plt.scatter(q, quantiles_exact - quantiles_approx)
plt.plot([0, 1], [0, 0])

# repeat it a few times and see what happens

## (b)


# confidence intervals for quantiles with bootstrap
# let us do this in two ways:
# (1) manualy : sample bootstrap samples in loop and calculate quantiles
# (2) scipy.stats.bootstrap

# (1)
N = 10_000
qs = []

for i in range(N):
    x_boot = np.random.choice(x, size = len(x), replace = True)
    qs.append(np.quantile(x_boot, q = q))

# instead of a list we better use np.array
qs = np.vstack(qs)

# now we can calculate mean quantile values over all N bootstrap samples
qs.mean(axis = 0)

# let us now find 95% confidence intervals for those N samples (+median)
pd.DataFrame(np.quantile(qs, axis = 0, q = np.r_[0.025, 0.5, 0.975]))

# and instead of just confidence intervals, let us plot boxplots for 
# each considered quantile value
pd.DataFrame(qs, columns=q).boxplot(flierprops={'marker': 'o', 
                                                'markersize': 3, 
                                                'markerfacecolor': 'grey'})
plt.scatter(range(1, len(q)+1), quantiles_exact, marker="x", c="red")

np.quantile(qs, axis = 0, q = np.r_[0.025, 0.975])

fig, ax = plt.subplots()
fig.dpi = 500
ax.scatter(quantiles_exact, q, s=30, marker='x', label = 'exact')

ax.errorbar(quantiles_approx, q, 
            xerr = np.quantile(qs, axis = 0, q = np.r_[0.025, 0.975]),
            fmt = 'o', # type of line connecting points
            markerfacecolor = 'darkred',
            markeredgecolor = 'black',
            markersize = 4,
            ecolor = 'black',
            elinewidth = 1,
            capsize = 3,
            zorder = -1,
            label = 'Estimated')
plt.xlabel('Confidence interval (95%)')
plt.ylabel('Quantile')
plt.legend(loc = 'lower right')

# (2)
import scipy.stats
boot = scipy.stats.bootstrap(
    (x,), 
    lambda x : np.quantile(x, q = q), 
    confidence_level = 0.95,
    n_resamples = N)

np.array(boot.confidence_interval)
np.quantile(qs, axis = 0, q = np.r_[0.025, 0.975])


fig, ax = plt.subplots()
fig.dpi = 500
#ax.scatter(quantiles_approx, q, s=30, marker='o')
ax.scatter(quantiles_exact, q, s=30, marker='x', label = 'exact')

ax.errorbar(quantiles_approx, q, 
            xerr = np.quantile(qs, axis = 0, q = np.r_[0.025, 0.975]),
            fmt = 'o', # type of line connecting points
            markerfacecolor = 'darkred',
            markeredgecolor = 'black',
            markersize = 4,
            ecolor = 'black',
            elinewidth = 1,
            capsize = 3,
            zorder = -1,
            label = 'Estimated manually')
ax.errorbar(quantiles_approx, q + 0.005, 
            xerr = np.array(boot.confidence_interval),
            fmt = 'o', # type of line connecting points
            markerfacecolor = 'darkred',
            markeredgecolor = 'black',
            markersize = 4,
            ecolor = 'blue',
            elinewidth = 1,
            capsize = 3,
            zorder = -1,
            label = 'Estimated scipy.stats.bootstrap')
plt.xlabel('Confidence interval (95%)')
plt.ylabel('Quantile')
plt.legend(loc = 'lower right')



