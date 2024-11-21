'''
Statistics and Data Analysis
Solutions to L06_ilumination.pdf
See also:
https://www.ibspan.waw.pl/~opara/statistics_and_data_analysis/DiffractionImages.pdf
'''



import numpy as np
from PIL import Image, ImageOps
 
# read image: https://www.ibspan.waw.pl/~opara/res/i1s.jpg
img = Image.open('i1s.jpg')
img.show()
 
# learn how to convert between image and np.ndarray

# what is the shape if this array?

# what does each layer represent?

# how to convert it to gray scale?
# - do it with ready function and manually: average values of layers




# (a)
# how to calculate means in each columns of an array?


# (b)
# how to subtract a value from each column?

# and how to do it if the task was to subtract not colum mean but row means?

# (c)
# the computed means should not take into account small and large values of
# colors. question: which are small and large?
# later subtract only from values which were taken to caculate the means


# (d)
# comapre results obtained with (b) and (d)


# (e)
img = Image.open('i2s.jpg')
img = Image.open('i3s.jpg')
 
