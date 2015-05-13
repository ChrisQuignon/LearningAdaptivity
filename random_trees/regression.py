#!/usr/bin/env python

from imp import load_source
from sklearn.ensemble import RandomForestRegressor
from random import randrange
import numpy as np
import pylab

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

ds = helper.stretch(ds)

#subsampling
# ds = helper.subsample_hours(ds)
ds = helper.subsample_n_minutes(ds, 2)# do it, only takes a few minutes!


hour = 0.000176125 #percentage value of one hour for the whole dataset
test_percentage = 7 * 24 * hour# 0.000352315 = one hour

test_items = int(test_percentage*len(ds))

#pick a random testset
# for _ in range(test_items):
#     rand_elem = train_set.pop(randrange(len(train_set)))
#     test_set.append(rand_elem)

# take the last n values as testset
test_set = ds[-test_items:]
train_set = ds[:-test_items]

#OUTPUT
output_keys = ['Energie', 'Leistung', 'Volumenstrom']
# output_keys = ['Ruecklauftemperatur', 'Volumenstrom']

#looks a little complicated, mabye we just slice...
_, train_out = helper.split(train_set, output_keys)
train_out, _ = helper.as_matrix(train_out)

_, test_out = helper.split(test_set, output_keys)
test_out, _ = helper.as_matrix(test_out)

#TRAIN SET
#The output and the input don't have to be disjoint for prediction disjoint
train_set, _ = helper.split(train_set, []) #splits off the dates
train_set, _ = helper.as_matrix(train_set)

#TEST SET
#The output and the input don't have to be disjoint for prediction disjoint
test_set, _ = helper.split(test_set, []) #splits off the dates
test_set, _ = helper.as_matrix(test_set)


#train forest
#TODO: This is nonesense, because the value to predict is inside the input value
#TODO: shift values to predict
forest = RandomForestRegressor(n_estimators = 100)
forest = forest.fit(train_set, train_out)
output = forest.predict(test_set)

for i in range(output.shape[1]):
    pylab.figure(figsize=(20,10)) #King size output
    pylab.title('Regression for ' + helper.translate(output_keys[i]) )
    #TODO: check if title is right
    pylab.plot(output[:, i], color = 'blue')
    pylab.plot(test_out[:, i], color = 'red')

    pylab.tight_layout()
    pylab.savefig('../img/regression-' + str(int(test_percentage/hour))+ helper.translate(output_keys[i]) + '.png')
    # pylab.show()

print forest.score(test_set, test_out)
print 'zu schoen um wahr zu sein...'
