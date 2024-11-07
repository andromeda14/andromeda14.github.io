"""
Solutions / hints to 'L05_correlations.pdf'
"""


import numpy as np
import pandas as pd
import scipy.stats

import sklearn.datasets
import matplotlib.pyplot as plt


iris = sklearn.datasets.load_iris(return_X_y = True, as_frame = True)

type(iris)

X = iris[0]
y = iris[1]

data = pd.concat([X, y], axis = 1)

# matplotlib tutorial
# https://matplotlib.org/stable/tutorials/pyplot.html

plt.scatter(X.iloc[:,0], X.iloc[:,1])
plt.scatter(X.iloc[:,0], X.iloc[:,1], c = y)

# lines and points options for plt.plot:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D




# manuall pairplot
fig, axs = plt.subplots(4, 4)
fig.dpi = 500
fig.set_figwidth(10)
fig.set_figheight(10)

for i in range(4):
    for j in range(4):
        if i == j:
            axs[i, i].hist(X.iloc[:, i])
        else:
            axs[i, j].scatter(X.iloc[:, j], X.iloc[:, i], 
                              c = y+1, marker='o', alpha=0.5,
                              s = 10, cmap='viridis')
        axs[i, j].set_xlabel(X.columns[j])
        axs[i, j].set_ylabel(X.columns[i])
        axs[i, j].label_outer()
        axs[i, j].tick_params(axis='both', labelsize=10)
        

# how to find correlations?
# see:
# np.corrcoef() (Pearson)
# pd.DataFrame.corr() (Pearson)
# scipy.stats.pearsonr (Pearson)
# scipy.stats.spearmanr (Spearman)
# scipy.stats.kendalltau (Kendall)

np.set_printoptions(suppress=True, linewidth=150, precision = 2)
# https://pandas.pydata.org/pandas-docs/dev/user_guide/options.html
pd.options.display.precision = 2
pd.options.display.width = 150
pd.options.display.max_columns = 5


# correlations for all columns:
r = X.corr()
np.corrcoef(X.T)
rho = scipy.stats.spearmanr(X)[0] # [1] p-value
tau = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        tau[i, j] = scipy.stats.kendalltau(X.iloc[:,i], X.iloc[:,j])[0]

r
pd.DataFrame(rho, columns=X.columns, index=X.columns)
pd.DataFrame(tau, columns=X.columns, index=X.columns)

pd.DataFrame(rho, columns=X.columns, index=X.columns) / \
    pd.DataFrame(tau, columns=X.columns, index=X.columns)
# rho (Spearman) tends to be larger than tau (Kendall)

# what can we say about those correlations?
# e.g. do longer sepals tend to be wider or narrower?

# how will it change when we take into account grouping into species?
data.groupby('target').corr()

# see Simpson paradox


# Pearson correlation - linear dependence only!

"""
"problems" with correlation
    - linear correlation: only linear dependence
       https://en.wikipedia.org/wiki/Correlation#/media/File:Correlation_examples2.svg
    - spurious correlations
      e.g https://www.datasciencecentral.com/spurious-correlations-15-examples/
    - Simpson's paradox 
      https://en.wikipedia.org/wiki/Simpson%27s_paradox
"""

def print_corrs(x, y):
    p = scipy.stats.pearsonr(x, y)[0]
    s = scipy.stats.spearmanr(x, y)[0]
    k = scipy.stats.kendalltau(x, y)[0]
    print(f'Pearson:  {p:.2f}\nSpearman: {s:.2f}\nKendall:  {k:.2f}')

x = np.linspace(0, 3*np.pi, 50)

y = np.sin(x)
plt.scatter(x, y)
print_corrs(x, y)
# there is clearly a dependence between x and y but its neither linear nor 
# ever monotonous

y = x**4
plt.scatter(x, y)
print_corrs(x, y)
# now, linear correlation is not high, but we can observe monotonous relation

x1 = np.random.uniform(0, 10, 50)
x2 = np.random.uniform(-10, 0, 50)
y1 = x1 + np.random.normal(0, 1, x1.shape[0])
y2 = x2 + 20 + np.random.normal(0, 1, x2.shape[0])
x = np.r_[x1, x2]
y = np.r_[y1, y2]
plt.scatter(x, y)
print_corrs(x, y)
print_corrs(x1, y1)
print_corrs(x2, y2)
# here we see different dependence in groups than in the whole dataset



# this time we can do necessary plot easily with seaborn
import seaborn as sns

    
sns.pairplot(X)
sns.pairplot(data, hue = 'target', palette = 'Set1')

# plot additionally regression line on whole set and within groups
sns.pairplot(X, kind="reg")
sns.pairplot(data, hue = 'target', palette = 'Set1', kind = 'reg')

