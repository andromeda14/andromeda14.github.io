import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

###############################################################################
# 1.2.1

import numpy as np
np.random.seed(13)
x = np.random.normal(0, 2, 20).round(2)

# (a)
x_abs = np.abs(x)
x[(x_abs > 1) & (x_abs < 2)]
                                     

# (b)
(x >= 0).sum()
(x >= 0).mean()

# (c)
x[x > 0].mean()

# (d)
x[np.argmin(x_abs)]

# (e)
x_greater = x < x.mean()

x1, x2 = x[x_greater], x[~x_greater]

# (f)

x = x - x.mean()
# x -= x.mean()
x /= x.std(ddof=1)

x.mean()
x.std(ddof=1)




###############################################################################
# 1.2.2

import numpy as np
np.random.seed(13)
x = np.random.normal(0, 2, 20).reshape(4, 5)


# see: shape broadcasting

# (a)
# standardise - mean should be eual to 0, and standard deviation eqial to 1
x -= x.mean(axis=0)
x /= x.std(axis=0, ddof=1)

# (b)
x -= x.mean(axis=1).reshape(-1, 1)
x /= x.std(axis=1, ddof=1).reshape(-1, 1)

# (c)
x.max(axis=1)

# (d)
x = np.exp(x)

x /= x.sum(axis=1).reshape(-1, 1)

# (e)
# only values smaller than 1, so max val
x.argmax(axis=1)


###############################################################################
# 1.2.3

n = 4
d = 5
x = np.random.normal(0, 1, n*d).reshape(n, d)
y = np.random.normal(0, 1, d)

np.sqrt(((x - y)**2).sum(axis=1))


###############################################################################
# 1.2.4
import pandas as pd

data = pd.read_csv('grades.csv', sep=";")
data.dtypes

data = data.set_index(['school', 'class'])
data = data.reset_index()

# (a)
data.loc[(data.school == "SP1") & (data.loc[:,"class"] == '2b'), :]

# (b)
data.loc[:, ['math', 'eng']].mean(axis=0)

# (c)
means = data.loc[:, ['class', 'math', 'eng', 'pe']] \
            .groupby('class').mean()
            
means.index            
means.columns

means = means.reset_index()

# (d)
means.sort_values('math')

# (e)
data.loc[:, 'above_school_average'] = np.where(data.math > data.math.mean(),
                                               "+", "-")

'''
# the same on groups - if student is above average in their class
tmp = data[['class', 'math']] \
    .groupby('class') \
    .apply(lambda x : x > x.mean()) \
    .reset_index()

data = pd.merge(data, tmp, left_on='student_id', right_on='level_1')
data.head()
'''

###############################################################################
# 2.1.1
# (a) 
t = np.linspace(-4, 4, 100)

x_norm = scipy.stats.norm(0, 1).pdf(t)
x_cauchy = scipy.stats.cauchy().pdf(t)

plt.plot(t, x_norm, label='norm')
plt.plot(t, x_cauchy, label='cauchy')
plt.legend()
# they seem similar, but they are not in fact
# cauchy does not have even first moment, has 'fat' tails







t = np.linspace(-10, 10, 100)
x_norm = scipy.stats.norm(0, 10).pdf(t)
x_cauchy = scipy.stats.cauchy().pdf(t)
plt.plot(t, x_norm, label='norm')
plt.plot(t, x_cauchy, label='cauchy')
plt.legend()


t = np.linspace(-50, 50, 100)
x_norm = scipy.stats.norm(0, 10).pdf(t)
x_cauchy = scipy.stats.cauchy().pdf(t)
plt.yscale('log')
plt.plot(t, x_norm, label='norm')
plt.plot(t, x_cauchy, label='cauchy')
plt.legend()


# (b) 
t = np.linspace(0, 30, 100)

shape = 30
scale = 0.5


x_norm = scipy.stats.norm(shape*scale, np.sqrt(shape*scale**2)).pdf(t)
x_gamma = scipy.stats.gamma(a=shape, scale=scale).pdf(t)

plt.plot(t, x_norm)
plt.plot(t, x_gamma)

###############################################################################
# 2.1.2

n = 20
p = 0.25
k = np.array([0, 5, 10, 15, 20])
scipy.stats.binom(n, p).pmf(k).round(4)


import matplotlib.pyplot as plt
k = np.arange(0, 21)
x = scipy.stats.binom(20, 0.25).pmf(k)


plt.scatter(k, x)

