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


def slice(ds, timedelta_input, timedelta_output, to_predict, freq=1):
    """
    Slices a dataframe into inputs and putputframes and ruturns them along with their shape.
    """

    inputs = []
    outputs = []
    input_shape= []
    output_shape = []

    last_time = ds.index[-2] - (timedelta_input + timedelta_output)

    print last_time

    for idx in range(0, ds.index.shape[0], freq):
        start_input_frame = ds.index[idx]
        end_input_frame = start_input_frame + timedelta_input
        end_output_frame = end_input_frame+timedelta_output

        input_frame = df[start_input_frame:end_input_frame]
        output_frame = df[end_input_frame:end_output_frame]

        for k in output_frame.keys():
            if k not in to_predict:
                del output_frame[k]

        inputs.append(input_frame.as_matrix().flatten())
        outputs.append(output_frame.as_matrix().flatten())

        input_shape = input_frame.shape
        output_shape = output_frame.shape

    return ((np.vstack(inputs), input_shape), (np.vstack(outputs), output_shape))



#split off the last timedelta as test data

timedelta_input = timedelta(hours = 3)
timedelta_output =  timedelta(hours = 1)
to_predict = ['Energie', 'Leistung']

last = df.index[-1] - timedelta_input - timedelta_output

#cutting off the last 4 hours is probably too little to predcit
train_frame = df[:last]
validation_frame = df[last:]

#sampling the "function"
(inputs, input_shape), (outputs, output_shape) = slice(train_frame, timedelta_input, timedelta_output, to_predict, 10)

inputs = np.asarray(inputs)
outputs = np.asarray(outputs)

print 'start learning', datetime.now()
forest = RandomForestRegressor(n_estimators = 10)
forest = forest.fit(inputs, outputs)

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

validation_out = np.reshape(validation_out, output_shape)
predicted_output = np.reshape(predicted_output, output_shape)

print 'validation:'
print validation_out.T
print ''
print 'prediction:'
print predicted_output.T
# print ''
# print forest.score([validation_in], [validation_out])

#TODO:
#Find optimal shift


pylab.plot(validation_out[:,0], color = "green", linestyle = '-')
pylab.plot(validation_out[:,1], color = "blue", linestyle = '-')
pylab.plot(predicted_output[:,0], color = "green", linestyle = '--')
pylab.plot(predicted_output[:,1], color = "blue", linestyle = '--')
pylab.show()
