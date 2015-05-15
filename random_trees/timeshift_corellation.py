#!/usr/bin/env python

from imp import load_source
from numpy import corrcoef, sum, log, arange, asarray, convolve, argmax, roll
from random import randrange
from pylab import pcolor, show, colorbar, xticks, yticks, tight_layout, plot, clf, title, xlim, savefig
import pandas as pd
import statsmodels.api as sm #needs R installed
from scipy.signal import correlate

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

#IMPORT
df = pd.DataFrame(ds)
df.set_index(df.Date, inplace = True)
df.interpolate(inplace=True)
df = df.resample('1Min')
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill


#FIND 1Day SEASONS

day = 60*24

dayseasons = {}

for key in df.keys():
    decomp = sm.tsa.seasonal_decompose(df[key].as_matrix(), freq=day)
    dayseasons[key] = decomp.seasonal[:day]

shifts = {}
for first in dayseasons.keys():
    shifts[first] = {}
    for second in dayseasons.keys():
        c = correlate(dayseasons[first], dayseasons[second])

        #find abs max
        c = c/max(abs(c))
        shift = (c.shape[0]/2 - argmax(c))/2

        shifts[first][second] = shift

        title(first + " - " + second + "\n" + "Shift: " + str(shift) + ' minutes')
        plot(c/max(c), color = "red")
        plot(dayseasons[first]/max(dayseasons[first]), color = "blue")
        plot(dayseasons[second]/max(dayseasons[second]), color = "green")
        xlim(0, c.shape[0])
        tight_layout()
        savefig('../img/timeshift-' + helper.translate(first) + '-' + helper.translate(second) + '.png')
        # show()
        clf()

# #VISUALISATION

sf = pd.DataFrame(shifts)
print sf

cm = sf.as_matrix()
key_order = asarray(['Aussentemperatur','Energie','Leistung','Niederschlag','Relative Feuchte','Ruecklauftemperatur','Volumenstrom','Vorlauftemperatur'])

# #REORDER
sortkey = asarray([7, 5, 0, 3, 4, 6, 2, 1])

cm = cm[sortkey]
for i,  row in enumerate(cm):
    cm[i] = row[sortkey]

pcolor(sf, cmap='RdBu')
cbar = colorbar()
cbar.set_label( 'shift in minutes')

lable = helper.translate(key_order[sortkey])

yticks(arange(0.5,8.5),lable, rotation='horizontal')
xticks(arange(0.5,8.5),lable, rotation='vertical')
tight_layout()
savefig('../img/timeshift.png')
# show()
clf()
