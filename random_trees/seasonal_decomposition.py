#!/usr/bin/python

from imp import load_source
from random import randrange, random
import numpy as np
from datetime import datetime, timedelta
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm #needs R installed


#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()



# ds = helper.stretch(ds)

df=pd.DataFrame(ds)
df.set_index(df.Date, inplace=True)
# df.interpolate(inplace=True)
df = df.resample('2Min')
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill
print(np.any(np.isnan(df.as_matrix())))


week = 30*24*7#Weekly season for two minutes
day = week/7

for key in df.keys():

    decomp = sm.tsa.seasonal_decompose(df[key].as_matrix(), freq=week)

    pylab.title(helper.translate(key))
    pylab.figure(figsize=(20,10)) #King size output
    decomp.plot()
    # pylab.show()

    pylab.tight_layout()
    pylab.savefig('../img/seasondecomposition-' + helper.translate(key) + '.png')
    pylab.clf()

    pylab.plot(decomp.seasonal[:week])
    pylab.title("Seasonal component for " + helper.translate(key))
    pylab.xticks(map(lambda x : x * day, range(7+1))) #one tick every day
    pylab.xlim(0, (week))
    pylab.xlabel("One week in steps of 2 minutes")

    pylab.tight_layout()
    # pylab.show()
    pylab.savefig('../img/season-' + helper.translate(key) + '.png')
    pylab.clf()
pylab.close()

print "this is awesome"
