"""
A short remainder to python, numpy and pandas.
It is not intended as a tutorial.
You can rather run code line by line, and check if the results are 
as you expected. 
If not: as me during class or look for an explanation.
"""

###############################################################################
# basic types
type(5)
type(5.0)
type(True)

5/2
type(5/2) # uff...


###############################################################################
# list - sequential, mutable, can store objects of different kinds

x = [True, 1, 8.5, 'a', lambda x : x**2, {'b' : 5}, [2, 3], (4, 5)]
type(x)
len(x)

[type(e) for e in x] # check the types of all elements

# indexing and slicing: also for tuples and strings
x = list(range(20))

x[0]
x[0] = 'other value'
x[-2] # x[len(x)-2]

x[2:5] # from:to (without 'to')
x[2:5] = ['some value']
x[2:]
x[:5]
x[2:9:2] # from:to:by
x[9:5:-1]
x[::-1] # reverse

x[:3] + x[:3]

type(x[0])
type(x[0:1])

x[(3, 7, 8)] # :(
[x[i] for i in [3, 7, 8]]


# list comprehension
x = []
for e in range(5):
    x.append(2*e)

[2*e for e in range(5)]

# operations on list are not vectorized
[1, 2, 3] + 2
[e + 2 for e in [1, 2, 3]]

'''
[fun(el) for el in iterable]
[fun(el) for el in iterable if cond(el)]
[fun1(el) if cond(el) else fun2(el) for el in iterable]
[fun1(el) if cond1(el) else fun2(el) for el in iterable if cond2(el)]
'''

[2 * e for e in range(5) if e > 1]

[2 * e if e < 3 else -1 for e in range(5)]

[2 * e if e < 3 else -3 * e for e in range(5) if e > 1]

# shoulde be used for simple tasks, it is not always faster than a regular 
# for loop, but for simple tasks it is more readible and faster to code

# import the following package and read ;)
import this # The Zen of Python, by Tim Peters

# some list methods (see more)
x.append(23)
x.extend([2, 3])
tmp = x.pop()
tmp = x.pop(4)
x.insert(3, 1)
x.count(1)
x.remove(1)


x = [3, 7, 1, 9, 0]
sorted(x)
x.sort() # inplace, returns None

# range - not a list
x = range(5)
type(x)

for e in x: print(e)
list(x)


###############################################################################
# tuple - sequential, immutable, can store objects of different types
# when function returns multiple objects it returns it in a tuple

def fun():
    return 1, 2

x = fun()
type(x)

x = (1, 'a', lambda x : x**2, {'b' : 5}, [2, 3], (4, 5))
type(x)
[type(e) for e in x]
# actually, what creates a tuple is not the brackets, but commas

x = 1, 2
type(x)
# one of the many reasons why a comma is not a good decimal point...


# indexing - same as list
x[0]
x[:3]
x[2](5)

# but... immutable
x[0] = 'other value'

# but
x = (1, 'a', lambda x : x**2, {'b' : 5}, [2, 3], (4, 5))

id(x)
id(x[4])
x[4][0] = 7 
id(x)
id(x[4])
# tuple x is not changed, its 5th element still points to the same list

# tuple comprehension? - no: generator
y = (e**2 for e in range(5))
print(y)
type(y)

for e in y: print(e)
for e in y: print(e) # now y is 'used'

###############################################################################
# dictionaries

d = {'a' : [1,2,3], 'b' : [4,5,6,7]}
type(d)

d[0] # not sequential (unordered elements)
d['a'] # by name
d.keys()
d.values()

# but we can iterate over a dict
for key in d:
    print(f'key: {key}, value: {d[key]}')

# dict comprehensions are also possible

###############################################################################
# strings - sequential

x = "Monty Python's Flying Circus"
x[2]
x[:5] + x[6:8]

# check some methods, e.g:
x.count('i')
x.split(" ")
x.endswith('us')
# and many more

###############################################################################
# some usefull functions

# enumerate
for i, e in enumerate(x):
    print(f'{i:2d}: {e}')


# zip
x = zip(range(5), ['a', 'b', 'c', 'd', 'e'])

for i, e in x:
    print(i, e)

list(x) # once again, it is 'used'


###############################################################################
# numpy arrays
import numpy as np

# numpy.ndarray is not a list and despite similarities there are significant
# differences

# creating an array
np.array([1, 2, 3])
np.arange(10)
np.linspace(3, 10, 10)
np.geomspace(3, 10, 10)
np.zeros(10)
np.r_[3, 5, 7]

# indexing and slicing - similar to lists, yet there are differences
x = np.arange(10)

# index with integers
x[0]
x[-1]

# slice x[start:stop:by]
x[2:7]
x[2:7:2]
x[2:7:-1]
x[7:2:-1]
x[1:]
x[:-2]
x[::-1]

# warning: slice returns a view (for list it was a copy)
y = x[:3]
y[0] = 100
x

x_list = x.tolist()
y_list = x_list[:3]
y_list[0] = 100
x_list
# in this case using array as if it was a list can lead to errors
# which are very hard to find

x = np.arange(10)
y = x[:3].copy() 
y[0] = 100
x


# vector of integers or list of integers (multiple selection possible)
x[[2,3,2,3]]
x[np.array([3,4,3,4])]

# many operations and functions are vectorised
x = np.arange(10)
y = np.arange(10, 20)

x + y # (no need for a loop)
x + 5
x**(1/y)
x > 5


# indexing with logical vector (very handy in data processing)
x[np.tile([True, False], 5)]
x[x > 5]

# 'and' and 'or' are not vectorized, we need other operators '|' and '&'
x[(x < 3) | (x > 6)]


# matrices - arrays of more dimensions
x = np.array([[1,2], [3,4]])
x.shape
np.zeros((2, 3))
np.r_[[1,2], [2,3]]
np.c_[[1,2], [2,3]]


# indexing and slicing
x = np.arange(12).reshape(3, -1)
x.shape

x[0, 1]
x[:2, :2]
y = x[:2, :2] # view!
y[0, 0] = 100
x


x[:, 0]
x[:, 0].shape

x[:, [0]]
x[:, [0]].shape

x[0,:]
x[[0],:]


# also:
x[[1, 2], [1, 2]] 
x[np.r_[1, 2], np.r_[1, 2]]

# but:
x[np.r_[1, 2], np.r_[1, 2].reshape(2, 1)]



# reshaping
x.shape
x.reshape(4,3)
x.reshape(2, -1)
x.reshape(2, 3, 2) # 3 dimensional array

# operators and shape broadcasting
x = np.zeros(12).reshape(4, -1)
y = np.r_[1, 10, 100] # same as 1-row matrix

x.shape, y.shape
x + y

y = y.reshape(1, -1)
x.shape, y.shape
x + y

y = y.reshape(-1, 1)
x.shape, y.shape
x + y # not possible - wrong shapes

y = np.r_[1, 10, 100, 1000].reshape(-1, 1)
x.shape, y.shape
x + y 

# arrays must have the same shape or one dimension must be the same and the 
# other equal to 1
# 1-dimensional array - treated as 1-row matrix in broadcasting

x = np.arange(12).reshape(4, -1)
x.T
np.sum(x)
x.sum()
sum(x)
sum(x.T)
x.sum(axis = 0)
x.sum(axis = 1)

x.max(axis=1)
x.argmax(axis = 0)

# see maaany useful functions in numpy package

###############################################################################
# pandas DataFrames

import pandas as pd
import sklearn.datasets
import seaborn
data = sklearn.datasets.load_wine(return_X_y = True, as_frame = True)

type(data)
data[0]
data[1]
data = pd.concat([data[0], data[1]], axis = 1)

type(data)
data.dtypes
data.head()

data.shape
data.columns # 'column names' - Index or MultiIndex
data.index # 'row names' - Index or MultiIndex

# selecting columns or rows
data['alcohol']
type(data['alcohol'])
data['alcohol'].shape

data[['alcohol']]
type(data[['alcohol']])
data[['alcohol']].shape

data.alcohol
type(data.alcohol)

# Series (one dimensional) - one column of a DataFrame (two dimensional)
# Series and DataFrames have different methods. 
# see e.g. unique() and drop_duplicates()

data.iloc[:, 0] # .iloc - by column number
type(data.iloc[:, 0])

data.iloc[:, 0]
type(data.iloc[:, 0])

data.loc[:, 'alcohol'] # .loc - by index, i.e by column name
type(data.loc[:, 'alcohol'])

data.loc[:, ['alcohol']]
type(data.loc[:, ['alcohol']])


data.loc[:, ['alcohol', 'ash']]
type(data.loc[:, ['alcohol', 'ash']])

data.loc[:, np.array(['alcohol'])]
type(data.loc[:, np.array(['alcohol'])])


data.iloc[:, :2]
data.loc[:, :2] # Error

# the same goes for rows - you can use either row index or row number,
# however in this frame index on rows is just RangeIndex

# here we can also index with boolean arrays
data.loc[data.alcohol > 14.5, :]
data.loc[data.alcohol > 14.5, data.columns.str.startswith('a')]



# there are many methods that DataFrame provides, check them out
# see Index and MultiIndex both on rows and columns
# see aggregating functions which can be applied on columns
# check out groupby - for operations on groups
# stack, unstack - columns to records and records to columns
# and more...


