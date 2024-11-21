'''
Statistics and Data Analysis
Solutions to L06_ilumination.pdf
'''

import numpy as np
from PIL import Image, ImageOps
 
# read image: https://www.ibspan.waw.pl/~opara/res/i1s.jpg
img = Image.open('i1s.jpg')
 
# converting between image and array
img.show()
data = np.array(img)
data.shape
#img2 = Image.fromarray(data) 
#img2.show()

# what is the shape if this array?
# what does each layer represent?

data1 = data.copy()
data1[:, :, [1, 2]] = 0
Image.fromarray(data1)

data3 = data.copy()
data3[:, :, [0, 1]] = 0
Image.fromarray(data3)

# how to convert it to gray scale?
# either use a ready function (it may produce 'better gray') or calculate 
# means in each layer


img_gray = ImageOps.grayscale(img)
data_gray = np.array(img_gray)
data_gray.shape

img_gray = Image.fromarray(data_gray) 

data_gray

# manually calculate means
data_gray_manual = data.mean(axis = 2).astype(np.uint8)
Image.fromarray(data_gray_manual)

# looks the same though
np.abs(data_gray - data_gray_manual).sum()


# (a)
# how to calculate means in each columns of an array?
b = data_gray.mean()
bj = data_gray.mean(axis=0)

import matplotlib.pyplot as plt
plt.plot(bj)

# (b)
# how to subtract a value from each column?

tmp = np.arange(12).reshape(3, 4)
tmp - tmp.mean(axis=0)


data_gray_fix1 = (data_gray - bj + b)
img_fix1 = Image.fromarray(data_gray_fix1)
img_fix1.show()

# (c)
# the computed means should not take into account small and large values of
# colors. question: which are small and large?

data_gray_mask = data_gray.copy()
data_gray_mask = data_gray_mask.astype(float)

mask = (data_gray_mask < 30) | (data_gray_mask > 220)

data_gray_mask[mask] = np.nan

b_robust = np.nanmean(data_gray_mask)
bj_robust = np.nanmean(data_gray_mask, axis = 0)


plt.plot(bj)
plt.plot(bj_robust)


# (d)
data_gray_fix2 = data_gray_mask - bj_robust + b_robust

data_gray_fix2[mask] = data_gray[mask]
img_fix2 = Image.fromarray(data_gray_fix2)
img_fix2.show()


# (e)
img = Image.open('i3s.jpg')
 
# converting between image and array
img.show()
data = np.array(img)
