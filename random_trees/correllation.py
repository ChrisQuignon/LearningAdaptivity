#!/usr/bin/env python

from imp import load_source
from numpy import corrcoef, sum, log, arange
from random import randrange
# import numpy as np
from pylab import pcolor, show, colorbar, xticks, yticks, tight_layout

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

ds = helper.stretch(ds)

values, _ = helper.split(ds, ['Date'])

matrix, keys = helper.as_matrix(values)

corrmatrix = corrcoef(matrix.T)

pcolor(corrmatrix)
colorbar()
yticks(arange(0.5,8.5),helper.translate(keys), rotation='horizontal')
xticks(arange(0.5,8.5),helper.translate(keys), rotation='vertical')
tight_layout()
show()
