'''
Solutions to tasks in Quantile_accuracy.pdf
'''

import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt

n = 50
shape = 3
rate = 0.5

# generate sample
# as always check which parameters you want to set and what kind of
# parameters are in scipy.stats

x = scipy.stats.gamma.rvs(a = shape, scale = 1/rate, size = 50)
plt.hist(x)

# calculate quantiles

q = np.r_[0, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1]
quantiles_approx = np.quantile(x, q)


quantiles_exact = scipy.stats.gamma.ppf(a = shape, scale = 1/rate, q = q)


# confidence intervals for quantiles with bootstrap
# let us do this with
# (a) manualy : sample bootstrap samples in loop and calculate quantiles
# (b) scipy.stats.bootstrap

# (a)
N = 10_000
qs = []

for i in range(N):
    x_boot = np.random.choice(x, size = len(x), replace = True)
    qs.append(np.quantile(x_boot, q = q))

# now qs is a list, but we can convert it into array
qs = np.vstack(qs)

# now we can calculate e.q means for each quatile
qs.mean(axis = 0)

# or plot thise distributions e.q with boxplots
pd.DataFrame(qs).boxplot()

# or confidence intervals:
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


# (b)
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
            xerr = np.array(boot.confidence_interval),
            fmt = 'o', # type of line connecting points
            markerfacecolor = 'darkred',
            markeredgecolor = 'black',
            markersize = 4,
            ecolor = 'black',
            elinewidth = 1,
            capsize = 3,
            zorder = -1,
            label = 'Estimated scipy.stats.bootstrap')
plt.xlabel('Confidence interval (95%)')
plt.ylabel('Quantile')
plt.legend(loc = 'lower right')





