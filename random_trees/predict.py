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
df = df.resample('10Min')
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



    for idx in ds.index:
        start_input_frame = idx
        end_input_frame = start_input_frame + timedelta_input
        end_output_frame = end_input_frame+timedelta_output

        input_frame = ds[start_input_frame:end_input_frame]
        output_frame = ds[end_input_frame:end_output_frame]

        for k in output_frame.keys():
            if k not in to_predict:
                del output_frame[k]

        if (input_shape == []):
            input_shape = input_frame.shape
            output_shape = output_frame.shape

        if (input_frame.shape == input_shape) and (output_frame.shape == output_shape):
            inputs.append(input_frame.as_matrix().flatten())
            outputs.append(output_frame.as_matrix().flatten())
        # else:
        #     print 'Frame dropped'
        #
        # print input_frame.shape
        # print output_frame.shape
        #

    return (inputs, input_shape), (outputs, output_shape)



#Define input and output frames
timedelta_input = timedelta(hours = 3)
timedelta_output =  timedelta(hours = 3)
to_predict = ['Energie', "Leistung"]

print 'Learning:'
print 'Input frame: ', timedelta_input
print 'Output frame: ', timedelta_output
print 'Features learned: ', helper.translate(to_predict)

#cutting off the validation frame
last = df.index[-1] - timedelta_input - timedelta_output
last = df.index[-1] - timedelta(days = 14)

train_frame = df[:last]
validation_frame = df[last:]

#sampling the "function"
(inputs, input_shape), (outputs, output_shape) = slice(train_frame, timedelta_input, timedelta_output, to_predict, 100)
(val_in, _), (val_out, _) = slice(validation_frame, timedelta_input, timedelta_output, to_predict, 100)

inputs = np.asarray(inputs)
outputs = np.asarray(outputs)
val_in = np.asarray(val_in)
val_out = np.asarray(val_out)

print 'start learning', datetime.now()
forest = RandomForestRegressor(n_estimators = 10, n_jobs = 8)
forest = forest.fit(inputs, outputs)
print 'stopped learning at ', datetime.now()

predicted_output = forest.predict(val_in)

#reshape rows according to output_shape
validation_out = np.reshape(val_out, (val_out.shape[0], output_shape[0], output_shape[1]))
predicted_output = np.reshape(predicted_output, (predicted_output.shape[0], output_shape[0], output_shape[1]))
score = forest.score(val_in, val_out)

print 'validation:'
print validation_out.T
print ''
print 'prediction:'
print predicted_output.T
print ''
print 'score:'
print score

#split values
val_features = np.dsplit(validation_out, validation_out.shape[2])
pred_features = np.dsplit(predicted_output, predicted_output.shape[2])

val_features = np.squeeze(val_features, axis=2)
pred_features = np.squeeze(pred_features, axis=2)

for i, v in enumerate(val_features):
    pylab.figure(figsize=(20,10))
    pylab.title(helper.translate(to_predict[i]))
    pylab.plot(np.ravel(val_features[i][:, 0], order='F'), marker="o")
    pylab.plot(np.ravel(pred_features[i][:, 0], order='F'), marker="x")
    in_h = timedelta_input.seconds /60/60
    out_h = timedelta_output.seconds /60/60
    pylab.savefig('../img/predict' + helper.translate(to_predict[i])+ '-' + str(in_h) + str(out_h) + '--' + str(score)[:4] + '.png')
    pylab.clf()
    # pylab.show()
