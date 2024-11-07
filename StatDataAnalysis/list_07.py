import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('lab_231117/brexit.csv',
                   index_col = False).iloc[:, 1:]
data.columns
data.dtypes

#convert to datetime format
date = pd.to_datetime(data.date, format="%Y-%m-%d")

# a

# plot series
plt.plot(date, data.loc[:, ['leave', 'stay']], 
         label = ['leave', 'stay'],
         linestyle='-')
plt.legend(loc='upper left', title='Decision')       


# plot points
plt.plot(date, data.loc[:, ['leave', 'stay']], 
         label = ['leave', 'stay'],
         linestyle='None', marker="o", alpha = 0.2)
plt.legend(loc='upper left', title='Decision')       


plt.plot(date, data.loc[:, ['leave', 'stay']], 
         label = ['leave', 'stay'],
         linestyle='-', marker="o", alpha = 0.4)
plt.legend(loc='upper left', title='Decision')       


# b

# compute moving average and add it to the plot
k = 10
ma = data.loc[:, ['leave', 'stay']].rolling(k).mean()

colors = ['tab:blue', 'tab:orange']
columns = ['leave', 'stay']

for column, color in zip(columns, colors):
    plt.plot(date, data[column], label = column,
             linestyle='None', marker="o", alpha = 0.2, color = color)
for column, color in zip(columns, colors):
    plt.plot(date, ma[column], color = color)
    
plt.legend(loc='upper left', title='Decision')       

# just the mooving averages
for column, color in zip(columns, colors):
    plt.plot(date, ma[column], color = color, label = column)
plt.legend(loc='upper left', title='Decision')       
# typical error/manipulation with plots: see the range on y-axis


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim(0, 55)
for column, color in zip(columns, colors):
    plt.plot(date, ma[column], color = color, label = column)
plt.legend(loc='lower left', title='Decision')       

# kernel smoothing instead of mooving average
ma_gaussian = data.loc[:, ['leave', 'stay']] \
                  .rolling(window=50, win_type="gaussian", center=True) \
                  .mean(std=5)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim(0, 60)
for column, color in zip(columns, colors):
    plt.plot(date, ma_gaussian[column], color = color, label = column)
plt.legend(loc='lower left', title='Decision')       

# window based on time interval, not number of variables
data2 = data.set_index('date')
data2.index = pd.to_datetime(data2.index)
ma_time = data2.loc[:, ['leave', 'stay']] \
                   .rolling(window='90D', center=False) \
                   .mean()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim(0, 60)
for column, color in zip(columns, colors):
    plt.plot(data2.index, ma_time[column], color = color, label = column)
plt.legend(loc='lower left', title='Decision')       


# to format axis nicely, see
# https://matplotlib.org/stable/api/dates_api.html
# https://stackoverflow.com/questions/60208121/matplotlib-display-only-years-instead-of-each-1st-january-in-x-axis-containing-d
import matplotlib.dates

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim(0, 50)

formatter = matplotlib.dates.DateFormatter("%Y")
locator = matplotlib.dates.YearLocator()
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)

for column, color in zip(columns, colors):
    ax.plot(date, ma[column], color = color, label = column)
ax.legend(loc='lower left', title='Decision')       
plt.show()


## c
leave = data['leave'].values
stay = data['stay'].values
date = date.values[1:]
delta_leave = leave[1:] - leave[:-1]
plt.plot(date, delta_leave, linewidth = 1)

## d
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(leave)
for k in range(1, 10):
    print('{1:02d}: {0:.2f}'.format(np.corrcoef(leave[k:], leave[:-k])[0,1],
                                    k))

plt.scatter(leave[1:], leave[:-1])
plt.scatter(leave[2:], leave[:-2])

# The autocorrelations are not big, which mean that series is to some extent 
# 'random'. 
# Predicting the next value basing on the current one is not easy

plot_acf(delta_leave)
for k in range(1, 10):
    print('{1:02d}: {0:.2f}'.format(np.corrcoef(delta_leave[k:], 
                                                delta_leave[:-k])[0,1],
                                    k))

plt.scatter(delta_leave[1:], delta_leave[:-1])
# negative first autocorrelation
# if delta_leave[i] > 0, then  delta_leave[i+1] is more likely to be < 0
# the series goes "up and down"


## e
k = 1
plt.scatter(leave[:-k], leave[k:])
plt.scatter(delta_leave[:-k], delta_leave[k:])
