#!/usr/bin/env python

from imp import load_source
from random import randrange, random
import numpy as np
from datetime import datetime, timedelta
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

# ds = helper.stretch(ds)

df = pd.DataFrame(ds)
df.set_index(df.Date, inplace = True)
df.interpolate(inplace=True)
df = df.resample('5Min')
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill


max_idx = df.index[-1] - timedelta(hours = 9)

def random_frameset(ds, timedelta_input, timedelta_output, to_predict):
    """Returns an input and an adjacent output frame with a given size
    example call:
    timedelta_input = timedelta(hours = 3)
    timedelta_output =  timedelta(hours = 1)
    to_predict = ['Energie', 'Leistung']
    in_frame, out_frame = random_frameset(ds, timedelta_input, timedelta_output, to_predict)"""

    max_time = df.index[-1] - timedelta_input - timedelta_output
    max_r = df[:max_time].shape[0]
    r = int(random() * max_r)

    start = df.index[r]
    now = start + timedelta_input
    prediction_goal = now+timedelta_output

    input_frame = df[start:(start+timedelta_input)]
    output_frame = df[now:prediction_goal]

    for k in output_frame.keys():
        if k not in to_predict:
            del output_frame[k]

    return input_frame, output_frame



#split off the last timedelta as test data

timedelta_input = timedelta(hours = 3)
timedelta_output =  timedelta(hours = 1)
to_predict = ['Energie', 'Leistung']

end = df.index[-1]
last = end - timedelta_input - timedelta_output

#cutting off the last 4 hours is probably too little to predcit
train_frame = df[:last]
validation_frame = df[last:]


#sampling the "function"
input = []
output = []
in_shape = []
out_shape = []
for _ in range(5000):
    in_frame, out_frame = random_frameset(train_frame, timedelta_input, timedelta_output, to_predict)

    in_shape = in_frame.shape
    out_shape = out_frame.shape

    in_matrix = in_frame.as_matrix().flatten()
    out_matrix = out_frame.as_matrix().flatten()
    #TODO: check for nan values here
    if not np.any(np.isnan(in_matrix)) and not np.any(np.isnan(out_matrix)):

        input.append(in_matrix)
        output.append(out_matrix)
        print "dopping frame"

input = np.asarray(input)#TODO check whether this is possible
output = np.asarray(output)

print 'start learning', datetime.now()
forest = RandomForestRegressor(n_estimators = 100)
forest = forest.fit(input, output)

print 'stopped learning at ', datetime.now()


#what to put in here?
start = validation_frame.index[0]
validation_in = validation_frame[start:start+timedelta_input]
validation_out = validation_frame[start+timedelta_input:start+timedelta_input+timedelta_output]
for k in validation_out.keys():
    if k not in to_predict:
        del validation_out[k]

validation_in = validation_in.as_matrix().flatten()
validation_out = validation_out.as_matrix().flatten()
predicted_output = forest.predict(validation_in)

validation_out = np.reshape(validation_out, out_shape)
predicted_output = np.reshape(predicted_output, out_shape)

print 'validation:'
print validation_out.T
print ''
print 'prediction:'
print predicted_output.T
# print ''
# print forest.score([validation_in], [validation_out])

#TODO:
#systematic evaluation
#Find optimal shift


pylab.plot(validation_out[:,0], color = "green", linestyle = '-')
pylab.plot(validation_out[:,1], color = "blue", linestyle = '-')
pylab.plot(predicted_output[:,0], color = "green", linestyle = '--')
pylab.plot(predicted_output[:,1], color = "blue", linestyle = '--')
pylab.show()
