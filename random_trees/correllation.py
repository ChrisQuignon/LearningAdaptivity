#!/usr/bin/env python

from imp import load_source
from numpy import corrcoef, sum, log, arange
from random import randrange
# import numpy as np
from pylab import pcolor, show, colorbar, xticks, yticks, tight_layout
import pandas as pd

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

df = pd.DataFrame(ds)

df = pd.DataFrame(ds)
df.set_index(df.Date, inplace=True)
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill

cm = df.corr()
pcolor(cm)
colorbar()

yticks(arange(0.5,8.5),helper.translate(cm.columns.values), rotation='horizontal')
xticks(arange(0.5,8.5),helper.translate(cm.columns.values), rotation='vertical')
tight_layout()
show()
