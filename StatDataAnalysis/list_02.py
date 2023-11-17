import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats



N = 100


# First, let us do the task for just one distribution and plot results
# to see what is happening on a simple data frame

# simple data frames for storing the results
results_mean = pd.DataFrame(np.nan, 
                            index = range(N), 
                            columns = [10, 100, 1_000, 10_000])


results_median = results_mean.copy()
results_trimmed = results_mean.copy()

for n in results_mean.columns:

    # random sample for given n and all N simulations simulnaneously
    
    x = np.random.normal(0, 1, (N, n)) # each row consists of 1 sample of length n
    #x = np.random.exponential(1, (N, n))
    #x = np.random.standard_cauchy((N, n))
    

    # calculating required statistics
    # for mean and median we can do it for a given dimension of a matrix
    # for scipy.stats.trim_mean we need to do it manually for each row     
    results_mean.loc[:, n] = x.mean(axis = 1) # mean for all samples
    results_median.loc[:, n] = np.median(x, axis = 1)
    for i in range(N):
        results_trimmed.loc[i, n] = scipy.stats.trim_mean(x[i,:], 0.1)
    

# now we can see how estimates change for different n and distributions

results_mean.boxplot(vert=False)

results_median.boxplot(vert=False)

results_trimmed.boxplot(vert=False)


# for cauchy, it is better to set limits for plotting to see what is
# happening

ax = results_mean.boxplot(vert=False)
ax.set_xlim((-10, 10))
plt.show()

# calculating mean and standard devaition

results_mean.mean(axis = 0)
results_mean.std(axis = 0)

# Now, we should repeat it for all distributions

# below is a compact solution to calculate everything that is needed


stats = ['mean', 'median', 'trimmed']
ns = [10, 100, 1_000, 10_000]
distrs = ['normal', 'exp', 'cauchy']

# create MultiIndex on columns
ind = pd.MultiIndex.from_product([stats, ns, distrs], 
                                 names = ['stat', 'n', 'distr'])

# creating empty frame
results = pd.DataFrame(np.nan, 
                       index = range(N), 
                       columns = ind)

for n in ns:
    for distr in distrs:
        if distr == 'normal': x = np.random.normal(0, 1, (N, n))
        elif distr == 'exp': x = np.random.exponential(1, (N, n))
        elif distr == 'cauchy': x = np.random.standard_cauchy((N, n))
        else: raise ValueError('incorrect distribution')
        
    
        results.loc[:, ('mean', n, distr)] = x.mean(axis = 1)
        results.loc[:, ('median', n, distr)] = np.median(x, axis = 1)
        for i in range(N):
            results.loc[i, ('trimmed', n, distr)] = \
                           scipy.stats.trim_mean(x[i,:], 0.1)
 
# aggregating: mean and standard deviation
results = results.agg(func = [np.mean, np.std], axis = 0)

# reshaping frame in needed way
results = results.stack(level=[0,2]) \
    .unstack(level=0) \
    .reorder_levels(axis=0, order = [1, 0]).sort_index(axis=0) \
    .reorder_levels(axis=1, order = [1, 0]).sort_index(axis=1)

# exporting data to csv
results.to_csv('results.csv')

# Now, what is left is to open it in Excel-like tool and format it in 
# needed way



