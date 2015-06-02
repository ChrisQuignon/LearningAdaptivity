#!/usr/bin/env python

from imp import load_source
from random import randrange, random
import numpy as np
from datetime import datetime, timedelta
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import csv

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

# ds = helper.stretch(ds)

df = pd.DataFrame(ds)
df.set_index(df.Date, inplace = True)
df.Energie.resample('1Min', fill_method="ffill")
df = df.resample('1Min')
# df.interpolate(inplace=True)
df.fillna(inplace=True, method='ffill')#we at first forwardfill
# df.fillna(inplace=True, method='bfill')#then do a backwards fill

#runs 1.5h on work laptop

def slice(ds, timedelta_input, timedelta_output, to_predict, stepwidth, input_sampling, output_sampling):
    """
    Slices a dataframe into inputs and outputframes and returns them along with their shape.
    The stepwidth of the sampling, the size of the frames and the sampling inside of the frames can be defined.
    """

    inputs = []
    outputs = []

    start_input_frame = ds.index[0]
    while start_input_frame +  timedelta_input + timedelta_output <= ds.index[-1]:

        end_input_frame = start_input_frame + timedelta_input
        end_output_frame = end_input_frame+timedelta_output

        input_frame = ds[start_input_frame:end_input_frame]
        output_frame = ds[end_input_frame:end_output_frame]

        input_frame = input_frame.resample(input_sampling)
        output_frame = output_frame.resample(output_sampling)

        for k in output_frame.keys():
            if k not in to_predict:
                del output_frame[k]

        inputs.append(input_frame.as_matrix().flatten())
        outputs.append(output_frame.as_matrix().flatten())

        #Move forward
        start_input_frame  = start_input_frame + stepwidth


    return (inputs, input_frame.shape), (outputs, output_frame.shape)


def wrapper(df, validation_delta, timedelta_input, timedelta_output, to_predict, input_sampling, output_sampling, stepwidth):
    """Wrapper for a complete prediction run on a given dataframe df"""

    print 'Learning:'
    print 'Input frame: ', timedelta_input
    print "Input sampling: ", input_sampling
    print 'Output frame: ', timedelta_output
    print "Output sampling: ", output_sampling
    print 'Stepwidth: ', stepwidth
    print 'Features to learn: ', helper.translate(to_predict)

    #CUTTING OF THE VALIDATION FRAME
    last = df.index[-1] - validation_delta

    train_frame = df[:last]
    validation_frame = df[last:]

    #SAMPLING THE FUNCTION
    (inputs, input_shape), (outputs, output_shape) = slice(train_frame, timedelta_input, timedelta_output, to_predict, stepwidth, input_sampling, output_sampling)
    (val_in, _), (val_out, _) = slice(validation_frame, timedelta_input, timedelta_output, to_predict, stepwidth, input_sampling, output_sampling)

    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)
    val_in = np.asarray(val_in)
    val_out = np.asarray(val_out)

    print 'start learning', datetime.now()
    forest = RandomForestRegressor(n_estimators = 10, n_jobs = 8)
    forest = forest.fit(inputs, outputs)
    print 'stopped learning at ', datetime.now()

    predicted_output = forest.predict(val_in)

    #RESHAPE AS BEFORE
    validation_out = np.reshape(val_out, (val_out.shape[0], output_shape[0], output_shape[1]))
    predicted_output = np.reshape(predicted_output, (predicted_output.shape[0], output_shape[0], output_shape[1]))
    score = forest.score(val_in, val_out)

    # print 'validation:'
    # print validation_out.T
    # print ''
    # print 'prediction:'
    # print predicted_output.T
    # print ''
    print 'score:'
    print score

    #EXPORT TO PNG
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
        pylab.savefig('../img/predict-' + helper.translate(to_predict[i])+ '-' + str(in_h) + str(out_h) + '--' + str(score)[:5] + '.png')
        # pylab.show()
        pylab.clf()

    return score


def testrun():
    """The place to set up a set of runs and write them to a csv file"""

    run = []

    #DROP PERCIPITATION
    ds = df.drop('Niederschlag', 1)

    keys = ["validation_delta",
            "timedelta_input",
            "timedelta_output",
            "to_predict",
            "input_sampling",
            "output_sampling",
            "stepwidth",
            "score",
            "runtime"]

    #PARAMETERS
    validation_delta = timedelta(days = 10)
    timedelta_inputs = [timedelta(hours = 24)]
    timedelta_output =  timedelta(hours = 24)
    to_predict = ['Energie', 'Leistung']
    input_sampling = '10Min'
    output_sampling = '10Min'
    stepwidths = [timedelta(minutes=10)]

    for timedelta_input in timedelta_inputs:
        for stepwidth in stepwidths:
            for _ in range(1):

                runtime = datetime.now()
                score = wrapper(ds, validation_delta, timedelta_input, timedelta_output, to_predict, input_sampling, output_sampling, stepwidth)
                runtime = datetime.now()- runtime

                d= {}
                d["validation_delta"] = validation_delta
                d["timedelta_input"] = timedelta_input
                d["timedelta_output"] = timedelta_output
                d["to_predict"] = to_predict
                d["input_sampling"] = input_sampling
                d["output_sampling"] = output_sampling
                d["stepwidth"] = stepwidth
                d["score"] = score
                d["runtime"] = runtime

                run.append(d)

    with open('../img/testrun.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(run)

#DO IT!
testrun()
