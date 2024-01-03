import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

###############################################################################
m = 1000
n = 50

pvals_shapiro = np.repeat(np.nan, m)
pvals_ks = np.repeat(np.nan, m)
pvals_ks_est = np.repeat(np.nan, m)

for i in range(m):
    x = np.random.normal(0, 1, n)
    pvals_shapiro[i] = scipy.stats.shapiro(x).pvalue
    pvals_ks[i] = scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(0, 1)).pvalue
    pvals_ks_est[i] = scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(x.mean(), x.std())).pvalue

# alpha = 0.05

(pvals_shapiro <  0.5).mean()
(pvals_shapiro <  0.05).mean()
(pvals_shapiro <  0.01).mean()

# as we can see the significance level for this test is indeed kept for any
# value of alpha

plt.hist(pvals_shapiro)

# does the same go for other tests?

alpha = 0.05
(pvals_ks <  alpha).mean()

# ok

(pvals_ks_est <  0.5).mean()
(pvals_ks_est <  0.05).mean()
(pvals_ks_est <  0.01).mean()


# it never rejects! Is it good then? 
# What about a situation when null hypothesis is not true?

m = 1000
n = 50

pvals_shapiro = np.repeat(np.nan, m)
pvals_ks_est = np.repeat(np.nan, m)
pvals_ks = np.repeat(np.nan, m)

for i in range(m):
    x = np.random.standard_t(1, n)
    pvals_shapiro[i] = scipy.stats.shapiro(x).pvalue
    pvals_ks[i] = scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(0, 1)).pvalue
    pvals_ks_est[i] = scipy.stats.kstest(x, scipy.stats.norm.cdf, args=(x.mean(), x.std())).pvalue

# How often, given significance level alpha = 0.05, test does not reject
# even if it should?

(pvals_shapiro > 0.05).mean()
(pvals_ks > 0.05).mean()
(pvals_ks_est > 0.05).mean()


1 - (pvals_shapiro > 0.05).mean()
1 - (pvals_ks > 0.05).mean()
1 - (pvals_ks_est > 0.05).mean()


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

from rpy2 import robjects

robjects.r("""
           x <- rnorm(10)
           print(sum(x))
           """)


robjects.r('''
           x <- rnorm(10)
           print(x)
           func <- function(x) {
               print("napis")}
           ''')

x_fromr = robjects.globalenv['x']
type(x_fromr)
np.array(x_fromr)


'''
Install R
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

see:
pwr::pwr.r.test

'''

# The necessary sample size to have 
# 80% probability of detecting correlation r = 0.3?
robjects.r('''
           result = pwr::pwr.r.test(r = 0.3, sig.level = 0.05, power = 0.8)
           str(result)
           ''')


# What correlation strengths are we likely to detect for sample size of n = 150?

robjects.r('''
           result = pwr::pwr.r.test(n = 150, sig.level = 0.05, power = 0.8) 
           str(result)
           ''')


# plot how will r and n depend for the fixed values of alpha and power

robjects.r('''
           r = seq(0.05, 0.95, 0.05)
           n = numeric(length(r))
           for(i in seq_along(r)){
                   n[i] = pwr::pwr.r.test(r = r[i], 
                                          sig.level = 0.05, 
                                          power = 0.8)$n}
           ''')

plt.plot(robjects.globalenv['n'], robjects.globalenv['r'])



'''
Check for your own: generate poorly correlated samples of diffent sizes 
and check how often the test detects this correlation.

scipy.stats.pearsonr
H0: no linear correlation
rejecting - there is a correlation

'''
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

n = 1_000
std = 10

x = np.linspace(0, 1, n)
y = x + np.random.normal(0, std, n)

plt.scatter(x, y)

scipy.stats.pearsonr(x, y)

# the bigger the sample the smaller correlation can be detected
# but is is always 'useful' correlation?

###############################################################################

# We not only want this effect to be present but we want the 
# meaningful size of the effect
# the Cohen d

n_x = x.shape[0]
n_y = y.shape[0]
var_x = np.var(x, ddof = 1)
var_y = np.std(y, ddof = 1)
s = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))

d = np.abs(x.mean() - y.mean()) / s

## (a)
# Warning! typo: treatment has mean 600, not 400

m_control = 500
sd_control = 400

m_treatment = 600
sd_treatment = 300

# Usually 500 +- 400, means that 400 is std. Here we will also assume that
# No sample sizes are given... (Life is hard!) 
# We will assume equal sizes, then s = sqrt((s_x^2 + s_y^2)/2)

s = np.sqrt((sd_control**2 + sd_treatment**2) / 2)
d = np.abs(m_control - m_treatment) / s
robjects.globalenv['d'] = robjects.FloatVector([d])

'''
Rule of thumb:
d approx 0.8 : large effect
d approx 0.5 : medium effect
d approx 0.2 : small effect
'''

## (b)
def find_cohen_d(x, y):
    n_x = x.shape[0]
    n_y = y.shape[0]
    var_x = x.var(ddof = 1)
    var_y = y.var(ddof = 1)
    s = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
    
    return np.abs(x.mean() - y.mean()) / s

n_pre = 30
x_control = np.random.normal(m_control, sd_control, n_pre)
x_treatment = np.random.normal(m_treatment, sd_treatment, n_pre)
x = x_control
y = x_treatment
d = find_cohen_d(x_control, x_treatment)
print(d)

## (c)
robjects.globalenv['d_est'] = robjects.FloatVector([d])

robjects.r('''
pow = pwr::pwr.t.test(d = d_est, 
                      power = 0.8, 
                      sig.level = 0.05,
                      type = "two.sample", 
                      alternative = c("greater"))
n_est = ceiling(pow$n)
print(n_est)

pow = pwr::pwr.t.test(d = d, 
                      power = 0.8, 
                      sig.level = 0.05,
                      type = "two.sample", 
                      alternative = c("greater"))
n = ceiling(pow$n)
print(n)
''')


