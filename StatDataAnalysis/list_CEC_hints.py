"""
Statistics and Data Analysis
Solutions to: 'Tests_CEC.pdf'
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import scipy.stats

data = pd.read_csv('https://www.ibspan.waw.pl/~opara/res/DECEC17.csv', 
                   sep = ';')
data.head()
data.dtypes



# (a)
# see data.columns.str.match() for regular expression


# (b)
# remind yourself how to extract columns and rows from a data frame
# see previous labs


# (c)
# divide plotting area:
fig, axs = plt.subplots(2, 2)
fig.dpi = 200

# now we can plot i (i, j) figure anything we want:
axs[i, j].hist()
axs[i, j].set_xlim()
axs[i, j].plot()
axs[i, j].scatter()
axs[i, j].axline()

# see documentation for parameters

# (d)
scipy.stats.shapiro()


# (e)
# Assumptions of KS-test: ?
# Is it a good idea to test normality with KS test?

scipy.stats.kstest()

# (f)
# Assumptions for t-test?

scipy.stats.ttest_ind()

# btw. boxplots here may be of use insteda of histograms
# it is also usefull to plot many boxplots on one plot
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
scipy.stats.ranksums()

# (h)

all_functions = data.Function.unique()
results = pd.DataFrame({'Function' : all_functions,
                        't' : np.repeat(np.nan, all_functions.shape[0]),
                        'ranksums' : np.repeat(np.nan, all_functions.shape[0])})

# try to fill this data frame




# (i)
# contingency table
pd.crosstab()



