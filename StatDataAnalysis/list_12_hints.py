import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

###############################################################################
# significance level, I-type error
# power of the test, II-type error

# Let us consider a sample x = (x_1, ...x_n) from N(0, 1)
# and 2 tests (3 actually):
# - Shapiro test for normality
# - Kolmogorov-Smirnov test for N(0, 1)
# - Kolmogorov-Smirnov test for N(x.mean(), x.std(ddof=1))

# We will test each sample with those 3 tests and get p-values
# Then we should repeat this procedure 10_000 times and get
# 10_000 p-values for each test

m = 10_000
n = 50

pvals_shapiro = np.repeat(np.nan, m)
pvals_ks = np.repeat(np.nan, m)
pvals_ks_est = np.repeat(np.nan, m)

for i in range(m):
    print(f"\r{i/m:.2f}", end = "")
    x = np.random.normal(0, 1, n)
    pvals_shapiro[i] = scipy.stats.shapiro(x).pvalue
    pvals_ks[i] = scipy.stats.kstest(x, scipy.stats.norm.cdf, 
                                     args=(0, 1)).pvalue
    pvals_ks_est[i] = scipy.stats.kstest(x, scipy.stats.norm.cdf, 
                                         args=(x.mean(), x.std(ddof=1))).pvalue


# Let us set significance level alpha=0.05
# What should we expect?
# How many of calculated p-values should be smaller than alpha?
# Once again - what is the significance level and I-type error?

# alpha <= P(reject H_0 | H_0 is true)
# now: H_0 is true (because we sampled from N(0, 1))
# and the probabilty of rejecting is:
(pvals_shapiro <  0.05).mean()


(pvals_shapiro <  0.01).mean()
(pvals_shapiro <  0.5).mean()

# as we can see the significance level for this test is indeed kept for any
# value of considered alpha, what we can also see here:

plt.hist(pvals_shapiro)

# does the same go for other tests?

(pvals_ks <  0.5).mean()
(pvals_ks <  0.05).mean()
(pvals_ks <  0.01).mean()

plt.hist(pvals_ks)

# seems fine. But this is not a situation we usually have in practice:
# we do not usually know EX and VarX - we need to estimate is from the sample

(pvals_ks_est <  0.5).mean()
(pvals_ks_est <  0.05).mean()
(pvals_ks_est <  0.01).mean()

plt.hist(pvals_ks_est)


# it almost never rejects! Is it good then? 
# Is significance level of the test kept?
# What can we say about the I-type error?
# How about the power of this test?
# Can we say anything about it from this experiment?



# What about a situation when null hypothesis is not true?
# Note: null hypothesis can be untrue in many, many ways!

# repeat the above task, but sample from different distribution
# e.g. x = np.random.standard_t(1, n)



# How often, given significance level alpha = 0.05, test does not reject
# even if it should?



# The situation will obviously change for different distribution of x
# Test behaviour for different sample sizes n : when power is higher? is 
# significance level kept always? and for different distributions

# P(reject | true H0) <= alpha
# P(not reject | true H1) = beta
# 1 - P(not reject | true H1) - power of the test

###############################################################################
'''
Tasks from L12_PowerAnalysis.pdf are based on R package 'pwr'. 
This is also related to your 2nd assignment.
We can run R code directly from python e.g. with 'rpy2' package
see e.g. https://rpy2.github.io/doc/v3.3.x/html/introduction.html
'''

# this package enables to run R code directly from python
# moreover we can use python variables in R code and R variables
# in our python code
from rpy2 import robjects

# simple example of R code
robjects.r("""
           x <- rnorm(10)
           print(sum(x))
           """)

print(x)
# :(

# see how to get to R variables
print(robjects.globalenv)
print(list(robjects.globalenv))
print(robjects.globalenv['x'])

robjects.r('''
           x <- rnorm(10)
           print(x)
           func <- function(x) {
               print("napis")}
           ''')

x = robjects.globalenv['x']
type(x)
np.array(x)

# and the other way around
y = np.random.normal(0, 1, 10)
robjects.globalenv['y'] = robjects.FloatVector(y)

robjects.r('''
           print(y)
           print(typeof(y))
           ''')

'''
If you are using google colab - package should be installed and it should work
Otherwise:
Install R (https://www.r-project.org/)
Install R package: pwr
See what functions are available
'''

robjects.r('''
           install.packages('pwr')
           ''')

robjects.r('''
           print(help(package = 'pwr'))
           ''')

robjects.r('''
           library(pwr)
           print(help(pwr.r.test))
           ''')


'''
https://cran.r-project.org/web/packages/pwr/vignettes/pwr-vignette.html
https://cran.r-project.org/web/packages/pwr/pwr.pdf
'''

###############################################################################
'''
Let us start with tasks (g) and (h)

(g) What is the necessary sample size to have 80% probability of detecting
correlation r = 0.3?

(What does it mean to detect a correlation?)

(h) What correlation strengths are we likely to detect for sample size of 
n = 150?
'''

## (g) and (h)
'''
H0: cor(x,y) = 0
H1: ~H0

In order to detect given correlation we need to reject the hypothesis.

P(reject | cor(x, y) == 0) <= alpha
P(do not reject | cor(x, y) !=0 ) = beta
P(reject | cor(x, y) != 0 ) = 1 - beta (power of the test)

but cor(x, y) != 0 can be not equal in many ways and beta will be different 
for different values

hint:
pwr::pwr.r.test()
and "Power Analysis" in lecture "06_design_of_experiments.pdf"
here, instead of Cohen d, we will input the value of correlation we want 
to detect: parameter r
'''



'''
Afterwards:
Check for your own: generate poorly correlated samples of diffent sizes 
and check how often test detect this correlation.
Plot some samples and see if such small correlation, even if detected,
can be always of use?

scipy.stats.pearsonr() - for correlation and also p-value
H0: no linear correlation
rejecting - there is a correlation

'''


# the bigger the sample the smaller correlation can be detected

###############################################################################


## (a)
# Achtung! typo: treatment has mean 600, not 400


# Usually 500 +- 400, means that 400 is std. Here we will also assume this
# No sample sizes are given... (Life is hard!) 
# We will assume equal sizes, then s = sqrt((s_x^2 + s_y^2)/2)


'''
Rule of thumb:
d approx 0.8 : large effect
d approx 0.5 : medium effect
d approx 0.2 : small effect
'''

## (b)

## (c)
# R function: pwr::pwr.t.test()
# it return results as named R list, to get element of a list by name:
# x <- list(a = 1:4, b = c('x, 'y'))
# x$a - element 'a'
# x['a'] - same



## (d)

## (e)

## (f)
# repeat everythinh for different values obtained in (b) or standard 
# deviations in (a)


