"""
Statistics and Data Analysis
Solutions to: 'Tests_CEC.pdf'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

data = pd.read_csv('DECEC17.csv', sep = ';')
data.head()
data.dtypes



# (a)
err_cols = data.columns.str.match('^Err.*')
data.loc[:, err_cols].head()

# (b)
x1 = data.loc[((data.loc[:, 'Algorigthm'] == 'DE_rand_1_Cr10') & (
           data.loc[:, 'Function'] == 27)), err_cols].values.flatten()

# or:
x1 = data.loc[((data.Algorigthm == 'DE_rand_1_Cr10') & (
           data.Function == 27)), err_cols].values.flatten()

x1 = data.loc[((data.Algorigthm == 'DE_rand_1_Cr10') & (
           data.Function == 27)), err_cols]
type(x1)

x1 = data.loc[((data.Algorigthm == 'DE_rand_1_Cr10') & (
           data.Function == 27)), err_cols].values
type(x1)
x1.shape

x1 = data.loc[((data.Algorigthm == 'DE_rand_1_Cr10') & (
           data.Function == 27)), err_cols].values.flatten()
x1.shape


x9 = data.loc[((data.loc[:, 'Algorigthm'] == 'DE_rand_1_Cr90') & (
           data.loc[:, 'Function'] == 27)) , err_cols].values.flatten()


# (c)
fig, axs = plt.subplots(2, 2)
fig.dpi = 200

x1_norm = x1 - x1.mean()
x1_norm /= x1_norm.std()

x9_norm = x9 - x9.mean()
x9_norm /= x9_norm.std()

tmp = np.r_[x1, x9]
x_limits = np.r_[tmp.min(), tmp.max()]

# (0, 0) - hist for x1
axs[0, 0].hist(x1, bins = 8, density=True)
axs[0, 0].set_xlim(x_limits)
t = np.linspace(x1.min(), x1.max(), 100)
axs[0, 0].plot(t, scipy.stats.norm.pdf(t, loc = x1.mean(), scale = x1.std()))

# (0, 1) - qqplot for x1 
qs = np.linspace(0, 1, 100)
axs[0, 1].scatter(scipy.stats.norm.ppf(qs), np.quantile(x1_norm, qs))
axs[0, 1].axline((0, 0), slope = 1, c = 'k')
# or simply sm.qqplot, but we can practive ploting in matplotlib

# (1, 0) - hist for x9
axs[1, 0].hist(x9, bins = 8, density = True)
axs[1, 0].set_xlim(x_limits)
t = np.linspace(x9.min(), x9.max(), 100)
axs[1, 0].plot(t, scipy.stats.norm.pdf(t, loc = x9.mean(), scale = x9.std()))

# (1, 1) - qqplot for x9 
qs = np.linspace(0, 1, 100)
axs[1, 1].scatter(scipy.stats.norm.ppf(qs), np.quantile(x9_norm, qs))
axs[1, 1].axline((0, 0), slope = 1, c = 'k')
# or simply sm.qqplot


# (d)
scipy.stats.shapiro(x1).pvalue
scipy.stats.shapiro(x9).pvalue
# when sample is small it is hard to reject hypothesis, but in case of x9 
# it is rejected

# (e)
# Is it a good idea to test normality with KS test?

scipy.stats.kstest(x1, scipy.stats.norm.cdf, args=(x1.mean(), x1.std())).pvalue
scipy.stats.kstest(x9, scipy.stats.norm.cdf, args=(x9.mean(), x9.std())).pvalue

# (f)

scipy.stats.ttest_ind(x1, x9).pvalue

# btw. boxplots here may be of use insteda of histograms
df = pd.DataFrame({'x' : np.r_[x1, x9],
                   'group' : np.repeat(['cr10', 'cr90'], 
                                       repeats = [x1.shape[0], x9.shape[0]])})

df.boxplot('x', by = 'group', vert=False, showmeans=True, 
           patch_artist = True,
           boxprops = dict(facecolor = "lightblue"),
           meanprops = dict(marker = 'x', 
                            markerfacecolor = 'black',
                            markeredgecolor = 'black'),
           medianprops = dict(color = 'black'))


# (g)
scipy.stats.ranksums(x1, x9).pvalue

# (h)

all_functions = data.Function.unique()
results = pd.DataFrame({'Function' : all_functions,
                        't' : np.repeat(np.nan, all_functions.shape[0]),
                        'ranksums' : np.repeat(np.nan, all_functions.shape[0])})

for i in range(results.shape[0]):
    x1 = data.loc[((data.loc[:, 'Algorigthm'] == 'DE_rand_1_Cr10') &
                   (data.Function == all_functions[i])), 
                  err_cols].values.flatten()
    x9 = data.loc[((data.loc[:, 'Algorigthm'] == 'DE_rand_1_Cr90') &
                   (data.Function == all_functions[i])), 
                  err_cols].values.flatten()
    
    t = scipy.stats.ttest_ind(x1, x9).pvalue
    ranksums = scipy.stats.ranksums(x1, x9).pvalue

    results.loc[i, ['t', 'ranksums']] = [t, ranksums] # index == row this time


results.set_index('Function')
results.loc[:, ['t', 'ranksums']] < 0.05
results.head()



# (i)
results2 = results.melt(id_vars = 'Function')
results2.loc[:, ['value']] = results2.loc[:, ['value']] < 0.05
pd.crosstab(index = results2['variable'], columns=results2['value'])


tmp = np.logical_xor(results.t.values < 0.05, ~(results.ranksums.values < 0.05))
tmp.sum(), (~tmp).sum()




