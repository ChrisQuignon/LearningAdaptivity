#!/usr/bin/env python

from imp import load_source
from numpy import corrcoef, sum, log, arange, asarray
from random import randrange
from pylab import pcolor, show, colorbar, xticks, yticks, tight_layout, plot, clf, title, xlim, savefig
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
cm = cm.as_matrix()
key_order = asarray(['Aussentemperatur','Energie','Leistung','Niederschlag','Relative Feuchte','Ruecklauftemperatur','Volumenstrom','Vorlauftemperatur'])

# #REORDER
sortkey = asarray([7, 5, 0, 3, 4, 6, 2, 1])

cm = cm[sortkey]
for i,  row in enumerate(cm):
    cm[i] = row[sortkey]

# #VISUALISATION

pcolor(cm, cmap='RdBu')
cbar = colorbar()

lable = helper.translate(key_order[sortkey])

yticks(arange(0.5,8.5),lable, rotation='horizontal')
xticks(arange(0.5,8.5),lable, rotation='vertical')
# tight_layout()
savefig('../img/correlation.png')
# show()
clf()
