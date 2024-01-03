import re
import numpy as np

import urllib.request

urllib.request.urlretrieve("https://www.ibspan.waw.pl/~opara/res/vc.r", "vc.txt")

# the file is given in a format which can be easily impotered to R,
# unfortunately for python there is a problem -> multiline string with 
# single quotes
# we can use any text editor to correct the file: " -> """
# after this fix and renaming to vc.py:

from vc import text_pl, text_en

text_pl
text_en


# you can also read file to python and then deal with it
# either way, after some effort we should have two variables
# text_pl
# text_en


## (a)
text_pl = re.sub("\W", "", text_pl) #non-word
text_en = re.sub("\W", "", text_en)

text_pl = text_pl.lower()
text_en = text_en.lower()


## (b)
text_en = re.sub("[aeiouy]", "1", text_en)
text_en = re.sub("[a-z]", "0", text_en)

text_pl = re.sub("[aąeęioóuy]", "1", text_pl)
text_pl = re.sub("\D", "0", text_pl) # non-digit

## (c)
text_pl = np.array(list(text_pl)).astype(int)
text_en = np.array(list(text_en)).astype(int)

# frequency of vowels
text_pl.mean()
text_en.mean()

# consonants
1 - text_pl.mean()
1 - text_en.mean()

## (d)
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(text_pl)
plot_acf(text_en)

np.corrcoef(text_pl[:-1], text_pl[1:])
np.corrcoef(text_en[:-1], text_en[1:])


## (e)
# to one string again:
text_en = "".join(text_en.astype(str))
text_pl = "".join(text_pl.astype(str))

en_distr = np.array([len(e) for e in text_en.split('1')])
pl_distr = np.array([len(e) for e in text_pl.split('1')])

import matplotlib.pyplot as plt
bins = np.arange(0.5, 6.5, 1)
plt.hist(en_distr[en_distr > 0], alpha = 0.5, density = True, label = 'en', bins = bins)
plt.hist(pl_distr[pl_distr > 0], alpha = 0.5, density = True, label = 'pl', bins = bins)
plt.legend()

# can those two distributions be considered equal?
# what kind of distributions is it
# what test can we use?

