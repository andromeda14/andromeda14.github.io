"""
Statistics and Data Analysis
Solutions to: 'L11_Tests.pdf'
"""

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# sample from given distributions
n = 50
x = np.random.normal(9, 1, n)
x.mean()

#x = np.random.standard_t(1, n) + 9
#x.mean()

#x = np.random.gamma(shape=2, scale=4.5, size=n)
#x.mean()

###############################################################################
## a

# plot histograms
fig = plt.figure(dpi = 500)
plt.hist(x, density = True)


# we can add density function to histograms, but with additional parameter
# density=True

fig = plt.figure(dpi = 1000)
plt.hist(x, density = True)

t = np.linspace(5, 13, 100)
plt.plot(t, scipy.stats.norm.pdf(t, 9, 1))


import statsmodels.api as sm 

# straight line, but is it an identity?
sm.qqplot(x)

# hmm... somethings's wrong...
sm.qqplot(x, line='45')

# here: loc - mean, scale - std
sm.qqplot(x, line='45', loc = 9, scale=1)
sm.qqplot(x, line='45', loc = x.mean(), scale=x.std())

# or
x_standardized = x - x.mean()
x_standardized /= x_standardized.std()

sm.qqplot(x_standardized, line='45')

# or 'manually': OX: theoretical quantiles, OY" empicical quantiles
qs = np.linspace(0, 1, 101)
plt.scatter(scipy.stats.norm.ppf(qs), np.quantile(x_standardized, qs))
plt.axline((0, 0), slope=1, c='k')

###############################################################################
## Tests
'''
Make sure that you know and understand notions:
- Type I error vs. Type II error
- Significance level: alpha
- Power of the test: 1 - beta
- Null hypothesis: H0 and Alternative hypothesis: H1
- Test statistic, critical region, p-value

Importance of assumptions to each test!
How statistical testing actualny works.

For answers, see, among others, notes from the lecture.
'''


###############################################################################
## b
"""
Assumptions: F is continuous.
 
H0: X ~ F
H1: ~H0

small p-value -> we reject H0, 
                 most probably X is not F distributed
big p-value -> we do not have reasons to reject H0, 
               there is not enough evidence to say that X is not F distr
"""

np.random.seed(123)
n = 50
x = np.random.normal(9, 1, n)

scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(9, 1)).pvalue
scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(9.5, 1)).pvalue
scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(8.5, 1)).pvalue
scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(9, 2)).pvalue

scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(9, 1)).pvalue
scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(x.mean(), x.std())).pvalue

# problem with KS Test?
# Need to estimate mean and std




###############################################################################
## c
'''
H0: X is normally distributed
small p-value -> we reject H0, 
                 most probably X is not normally distributed
big p-value -> we do not have reasons to reject H0, 
               there is not enough evidence to say that X is not normal
               
To test normality: shapiro is better               
'''

scipy.stats.shapiro(x).pvalue

###############################################################################
## d
'''
t-test:
H0: EX1 == EX2

small p-value -> we reject H0, 
                 cannot say that EX1 == EX2, 
                 most probably EX1 != EX2
big p-value -> we do not have reasons to reject H0, 
               there is not enough evidence to say that EX1 != EX2, so
               probably EX1 == EX2
               
warning: see the default value for equal_var. 
Can we assume equal variances here?
How can we test for equal variance?             
'''

x_norm = np.random.normal(9, 1, n)
x_t = np.random.standard_t(1, n) + 9

scipy.stats.ttest_ind(x_norm, x_t).pvalue

x_norm.std()
x_t.std()

#scipy.stats.bartlett(x_norm, x_t).pvalue


scipy.stats.ttest_ind(x_norm, x_t, equal_var=True).pvalue
scipy.stats.ttest_ind(x_norm, x_t, equal_var=False).pvalue


'''
How to test median?
H0: med(X1) == med(X2)

Usually we test distributions
'''

x_norm = np.random.normal(9, 1, n)
x_t = np.random.standard_t(1, n) + 9
scipy.stats.ranksums(x_norm, x_t).pvalue

###############################################################################
# e, f
'''
We can check more tests
'''

m = 100
n = 50
results = pd.DataFrame({'t' : np.repeat(np.nan, m),
                        't_diffvar' : np.repeat(np.nan, m),
                        'kruskal' : np.repeat(np.nan, m),
                        'ks' : np.repeat(np.nan, m),
                        'ranksums' : np.repeat(np.nan, m),
                        'mann' : np.repeat(np.nan, m)})


for i in range(m):
    print('\r{0:.2f}'.format(i/m), end='')

    x_norm = np.random.normal(9, 1, n)
    #x_t = np.random.normal(9, 1, n)
    x_t = np.random.standard_t(1, n) + 9

    results.loc[i, 't'] = scipy.stats.ttest_ind(x_norm, x_t).pvalue
    results.loc[i, 't_diffvar'] = scipy.stats.ttest_ind(x_norm, x_t, equal_var=False).pvalue
    results.loc[i, 'kruskal'] = scipy.stats.kruskal(x_norm, x_t).pvalue
    results.loc[i, 'ks'] = scipy.stats.kstest(x_norm, x_t).pvalue
    results.loc[i, 'ranksums'] = scipy.stats.ranksums(x_norm, x_t).pvalue
    results.loc[i, 'mann'] = scipy.stats.mannwhitneyu(x_norm, x_t).pvalue


results.head()


import seaborn as sns
sns.pairplot(results)




