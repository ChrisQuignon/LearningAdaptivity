#!/usr/bin/env python

from imp import load_source
from sklearn.ensemble import RandomForestClassifier
from random import randrange
import numpy as np

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

ds = helper.stretch(ds)

#make it short
ds = ds[0:1000]
#TODO: make it long



#pick training, test and validation data
train_set = ds
test_set = []
validation_set = []

test_percentage = 0.15
validation_percentage = 0.15

test_items = int(test_percentage*len(train_set))
validation_items = int(validation_percentage*len(train_set))

for _ in range(test_items):
    rand_elem = train_set.pop(randrange(len(train_set)))
    test_set.append(rand_elem)

for _ in range(validation_items):
    rand_elem = train_set.pop(randrange(len(train_set)))
    validation_set.append(rand_elem)




output_keys = ['Energie', 'Leistung']

train_in, train_out = helper.split(train_set, output_keys)
test_in, test_out = helper.split(test_set, output_keys)
validation_in, validation_out = helper.split(test_set, output_keys)


train_in, _ = helper.as_matrix(train_in)
train_out, _ = helper.as_matrix(train_out)


test_in, _ = helper.as_matrix(test_in)
test_out, _ = helper.as_matrix(test_out)


validation_in, _ = helper.as_matrix(validation_in)
validation_out, _ = helper.as_matrix(validation_out)


# # Create the random forest object which will include all the parameters
# # for the fit
forest = RandomForestClassifier(n_estimators = 100)
#
# # Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_in, train_out)
#
# # Take the same decision trees and run it on the test data
output = forest.predict(validation_in)

print (output -  validation_out)
print np.mean(output -  validation_out)
print 'yay, we predicted something'
