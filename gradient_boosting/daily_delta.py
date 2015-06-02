#! /usr/bin/python
# test_pandas.py
from imp import load_source
from random import randrange, random
import numpy as np
from datetime import datetime, timedelta
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from itertools import *
# import dataset
# equivalent to:
# import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

df = pd.DataFrame(ds)
df.set_index(df.Date, inplace=True)
df = df.interpolate()
df.fillna(inplace=True, method='ffill')  # we at first forwardfill
df.fillna(inplace=True, method='bfill')  # then do a backwards fill
df = df.resample('1Min', how="mean")
# df_norm = (df - df.mean()) / (df.max() - df.min())
# df = df_norm
df = df.dropna()
train_start = df.index.searchsorted(datetime(2014, 7, 1, 0, 0))
train_end = df.index.searchsorted(datetime(2014, 12, 1, 0, 0))


train_data = df.ix[train_start:train_end]


train_per = round(float(train_data.Aussentemperatur.count()) /
				  float(df.Aussentemperatur.count()) * 100, 2)


print("total data count: %d" % (df.Aussentemperatur.count()))

print("training data count: %d" % (train_data.Aussentemperatur.count()))


print("training on %.2f percent data" % (train_per))


to_be_predicted = ['Energie']
original_params = ["Aussentemperatur", "Relative Feuchte", "Ruecklauftemperatur",
			   "Volumenstrom", "Vorlauftemperatur"]


to_be_input = original_params


training_ip = train_data.loc[:, to_be_input].values
training_op = train_data.loc[:, to_be_predicted].values

#other parameter do not mater
regressor = RandomForestRegressor(n_jobs=100)

regressor = regressor.fit(training_ip, training_op.ravel())
regressor_name = str(regressor).split("(")[0]

MSE_list = []
test_start = df.index.searchsorted(datetime(2014, 12, 1, 0, 1))
days_limit = 10

score = regressor.score(training_ip, training_op)
print("Score: %.2f" % (score))

for step in range(1, days_limit + 1):
	print("-------------------------")
	test_end = df.index.searchsorted(
		datetime(2014, 12, 1, 0, 1) + timedelta(days=step))
	test_data = df.ix[test_start:test_end]
	test_per = round(float(
		test_data.Aussentemperatur.count()) / float(df.Aussentemperatur.count()) * 100, 2)
	print("testing data count: %d" % (test_data.Aussentemperatur.count()))
	print("testing on %.2f percent data" % (test_per))

	test_ip = test_data.loc[:, to_be_input].values
	test_actual = test_data.loc[:, to_be_predicted].values

	test_predictions = regressor.predict(test_ip)

	MSE = mean_squared_error(test_actual, test_predictions)**0.5
	R2 = r2_score(test_actual, test_predictions)
	mae = mean_absolute_error(test_actual, test_predictions)
	print("Mean Squared Error : %.2f" % (MSE))
	MSE_list.append(MSE)


	if step == days_limit:
		pylab.figure(figsize=(20,10))
		pylab.plot(test_predictions, color = 'red')
		pylab.plot(test_actual, color = 'green')
		pylab.title("Mean Squared Error : %.2f"%(MSE))
		pylab.xlabel("5 minutes")
		pylab.ylabel(helper.translate(to_be_predicted[0]))
		pylab.tight_layout()
		# pylab.show()
		pylab.savefig('../img/' + str(score) +regressor_name +'.png')
		pylab.clf()

pylab.figure(figsize=(20,10))
pylab.plot(MSE_list)
pylab.legend(["RMSE"])

pylab.title(str(to_be_input))
pylab.xlabel("time delta += 1 day")
pylab.ylabel("Error")
# ax = pylab.gca()
# ax.set_yticks(np.arange(20))
plt.grid()
pylab.savefig('../img/' + str(score) +regressor_name + '_day_error.png')
# pylab.show()
pylab.close()
