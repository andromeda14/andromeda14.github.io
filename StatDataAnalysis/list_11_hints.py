"""
Statistics and Data Analysis
Solutions to: 'L11_Tests.pdf'
"""

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# sample from given distributions
np.random.normal()
np.random.standard_t()
np.random.gamma()

# see documentation for parameters

###############################################################################
## a

# plot histograms
plt.hist()

# by default hist plots the number of observations in ech group
# to add density function to histograms, we need add additional parameter
# density=True

scipy.stats.norm.pdf() # density function for normal distribution


import statsmodels.api as sm 

sm.qqplot()
# plot qq-plot and
# straight line, but is it a identity?
# see documentation for more parameters - add identity line and 
# additional parameters for normal distribution


# plot also QQ plot 'manually'
# calculate theoretical and empirical quantiles and plt them:
# OX: theoretical quantiles, OY: empicical quantiles
plt.scatter()
plt.axline() # to plot straight line

###############################################################################
## Tests
'''
Make sure that you know and understand notions:
- Type I error vs. Type II error
- Significance level alpha
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

scipy.stats.kstest()

# problem with KS Test?
# nonparametric, very general, for any continuous distribution
# need to estimate mean and std
# small power




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

scipy.stats.shapiro()

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

scipy.stats.ttest_ind()

# can we assume that variance is equal in both samples?
# see additional parameters of this test

#scipy.stats.bartlett(x_norm, x_t).pvalue


'''
How to test median?
H0: med(X1) == med(X2)

Usually we test distributions
'''

scipy.stats.ranksums()

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

# check how each of those tests work: use them m times, each time
# for a sample of length n, and store p-values in 'results' data frame


results.head()



results.boxplot()

# do those tests keep the significance level?
alpha = 0.05
(results < alpha).mean(axis=0)


