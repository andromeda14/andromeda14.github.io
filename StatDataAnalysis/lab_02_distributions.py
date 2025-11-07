import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

"""
What do we already know:
    - Basic distributions and their densities or probability mass functions
    - how one can represent and distributions in python: scipy.stats
    - density: scipy.stats.xxx.pdf, pmf: scipy.stats.xxx.pmf
    - how to plot such density: 
        matplotlib.pyplot.plot or matplotlib.pyplot.scatter


For today:
    - what is the cumulative distribution function
    - what is quantile function
    - plotting histograms, boxplots, CDF and ECDF, qq-plots
    - tools for plotting: pandas, matplotlib, seaborn
"""


#############################################################################
# Plotting CDF
# scipy.stats.xxx.cdf
# matplotlib.pyplot.plot


# exponential
t = np.linspace(0, 10, 100)
lam = 0.5

x = scipy.stats.expon(scale=1/lam).cdf(t)
plt.plot(x)

for lam in [0.1, 0.5, 1, 2, 5]:
    x = scipy.stats.expon(scale=1/lam).cdf(t)
    plt.plot(x, label=lam)
plt.legend()



# normal
t = np.linspace(-4, 4, 100)

mean = 0
sigma = 1

x = scipy.stats.norm(mean, sigma).cdf(t)
plt.plot(t, x)

ax = plt.subplot(111)
for mean, sigma in zip([0, 0, 1], [1, 0.5, 1]):
    x = scipy.stats.norm(mean, sigma).cdf(t)
    ax.plot(t, x, label=f"({mean}; {sigma:.1f})")
ax.legend()

# binom
n = 20
p = 0.5
t = np.linspace(0, n, 1000)

x = scipy.stats.binom(n, p).cdf(t)
plt.plot(t, x)

ax = plt.subplot(111)
for p in [0.1, 0.5, 0.9]:
    x = scipy.stats.binom(n, p).cdf(t)
    ax.plot(t, x, label=p)
ax.legend()    
    
###############################################################################
mean = 0
sigma = 1
CDF_norm = scipy.stats.norm(mean, sigma).cdf

for a in [1, 3, 5, 7, 9]:
    result = CDF_norm(a) - CDF_norm(-a)
    print(f"CDF_norm({a}) - CDF_norm(-a) = {result:.5f}")

CDF_cauchy = scipy.stats.cauchy().cdf
for a in [1, 3, 5, 7, 9, 20, 100]:
    result = CDF_cauchy(a) - CDF_cauchy(-a)
    print(f"CDF_cauchy({a}) - CDF_cauchy(-{a}) = {result:.5f}")


###############################################################################

# for one sample
n = 100
x = scipy.stats.norm(0, 1).rvs(n)
# or with numpy

x = np.random.normal(0, 1, n)
x = np.random.uniform(-3, 3, n)

plt.boxplot(x, vert=False)
plt.violinplot(x, vert=False)

plt.hist(x)





# let us compare a few distributions on one plot
import pandas as pd
n = 100
rng = np.random.default_rng()

df = pd.DataFrame({'pareto' : rng.pareto(2, n),
                   'normal' : rng.normal(0, 1, n),
                   'uniform' : rng.uniform(-3, 3, n),
                   'cauchy' : rng.standard_cauchy(n)})
print(df)
df.shape
df.columns
df.index
df.dtypes

df.loc[:, ['normal']]
df.loc[:5, ['normal']]

# boxplot
df.boxplot(vert=False)

df.loc[:, ['uniform', 'normal']].boxplot(vert=False)

df.loc[:, ['uniform', 'normal', 'pareto']].boxplot(vert=False)

# violinplot
plt.violinplot(df, vert=False, showextrema=False, showmedians=True)
plt.violinplot(df.loc[:, ['normal', 'uniform']], 
               vert=False, showmedians=True)

# boxplot + violinplot
df.boxplot(vert=False)
plt.violinplot(df, vert=False, showextrema=False, showmedians=True)

# histograms
df.hist()

import seaborn as sns

# comaprinf two empirical distributions
sns.histplot(df, kde=True)
sns.histplot(df.loc[:, ['normal', 'cauchy']], kde=True,)

plt.xlim((-20, 20))
sns.histplot(df.loc[:, ['normal', 'cauchy']], kde=True,)

sns.histplot(df.loc[:, ['normal', 'uniform']], kde=True)
sns.histplot(df.loc[:, ['normal', 'pareto']], kde=True)

# normal: hist, kernel density estimator, true density
sns.histplot(df.loc[:, ['normal']], kde=True, stat='density')
t = np.linspace(-3, 3, 100)
plt.plot(t, scipy.stats.norm(0, 1).pdf(t), color='k')

# cauchy: hist, kernel density estimator, desity for normal
sns.histplot(df.loc[:, ['cauchy']], kde=True, stat='density')
t = np.linspace(-20, 20, 100)
plt.plot(t, scipy.stats.norm(0, 2.5).pdf(t), color='k')


# ECDF
x = df.loc[:, "normal"]
x = np.sort(x)

x = pd.DataFrame({'x' : x, 
                  'd' : np.arange(1, x.shape[0]+1) / x.shape[0]})

plt.plot(x.loc[:, 'x'], x.loc[:, 'd'])

ax = plt.subplot(111)
ax.set_xlim((-5, 5))
for name in df.columns:
    x = df.loc[:, name]
    x = np.sort(x)
    
    x = pd.DataFrame({'x' : x, 
                      'd' : np.arange(1, x.shape[0]+1) / x.shape[0]})
    
    ax.plot(x.loc[:, 'x'], x.loc[:, 'd'], label=name)
ax.legend()
    






a = 2
b = 5
n = 200
N = 1000
x = np.random.uniform(a, b, (N, n))
x[0, :]

x.mean()
x.mean(axis=0)
x.mean(axis=1)

plt.hist(x.mean(axis=1), density=True)

###############################################################################
# CLT

N = 1000
n = 200
a = 0
b = 1

x = np.random.uniform(a, b, (N, n))
mean = (b - a) / 2
sd = np.sqrt((b - a)**2 / 12)



plt.hist(x.mean(axis=1), density=True)
t = np.linspace(a, b, 100)
plt.plot(t, scipy.stats.norm(mean, sd/np.sqrt(n)).pdf(t))
#plt.plot(t, scipy.stats.norm(x.mean(axis=1).mean(), x.mean(axis=1).std()).pdf(t))
x.mean(axis=1).mean()
x.mean(axis=1).std(ddof=1)


###############################################################################
# quantiles

x = np.array([0, 0, 0, 0, 0, 1, 1, 8, 9, 9])
#x = np.random.normal(0, 1, 100) 
q = np.r_[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

np.quantile(x, q, method='inverted_cdf')

methods = ['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 
           'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 
           'median_unbiased', 'normal_unbiased', 'lower', 'higher', 
           'midpoint', 'nearest']


results = []
for m in methods:
    results.append(np.quantile(x, q, method=m))

df = pd.DataFrame(np.vstack(results), index=methods, columns=q)

df.T.plot()

ax = df.T.plot()
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.plot()

###############################################################################
# or 'manually': OX: theoretical quantiles, OY" empicical quantiles
# to understand what qqplot is

qs = np.linspace(0, 1, 101)

# (a)
x = np.random.normal(0, 1, 1000)
plt.scatter(scipy.stats.norm.ppf(qs), np.quantile(x, qs))
plt.axline((0, 0), slope=1, c='k')

# (b)
x = np.random.standard_cauchy(1000)
plt.scatter(scipy.stats.norm.ppf(qs), np.quantile(x, qs))
plt.axline((0, 0), slope=1, c='k')

# (c)
x = np.random.exponential(0.5, 1000)
plt.scatter(scipy.stats.norm.ppf(qs), np.quantile(x, qs))
plt.axline((0, 0), slope=1, c='k')




