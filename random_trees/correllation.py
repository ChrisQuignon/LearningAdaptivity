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
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill
df.set_index(df.Date, inplace=True)


cm = df.corr()


# #RESORT
# key_order = ['Aussentemperatur', 'Niederschlag', 'Relative Feuchte', 'Ruecklauftemperatur', 'Volumenstrom', 'Vorlauftemperatur', 'Energie', 'Leistung']
#
# sortkey = np.asarray([0, 1, 3, 5, 7, 2, 4, 6])
# cm = cm.as_matrix()
#
# cm = cm[sortkey]
#
# for row in cm:
#     row = row[sortkey]
#
# print cm



pcolor(cm)
colorbar()

yticks(arange(0.5,8.5),helper.translate(cm.columns.values), rotation='horizontal')
xticks(arange(0.5,8.5),helper.translate(cm.columns.values), rotation='vertical')
tight_layout()
show()
