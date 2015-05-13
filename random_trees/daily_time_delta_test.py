#! /usr/bin/python
#test_pandas.py
from imp import load_source
from random import randrange, random
import numpy as np
from datetime import datetime, timedelta
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()


df = pd.DataFrame(ds)
df.set_index(df.Date, inplace=True)
df = df.interpolate()
df.fillna(inplace=True, method='ffill')#we at first forwardfill
df.fillna(inplace=True, method='bfill')#then do a backwards fill
df = df.resample('5Min',how="mean")
df = df.dropna();
train_start = df.index.searchsorted(datetime(2014, 7, 1,0,0))
train_end = df.index.searchsorted(datetime(2014, 12, 1,0,0))


train_data = df.ix[train_start:train_end]


train_per = round(float(train_data.Aussentemperatur.count())/float(df.Aussentemperatur.count())*100,2)


print("total data count: %d"%(df.Aussentemperatur.count()))

print("training data count: %d"%(train_data.Aussentemperatur.count()))


print("training on %.2f percent data"%(train_per))


to_be_predicted = ['Energie'];
to_be_input = ["Aussentemperatur","Niederschlag","Relative Feuchte","Ruecklauftemperatur",
			   "Volumenstrom" , "Vorlauftemperatur"]
training_ip = train_data.loc[:,to_be_input].values;
training_op = train_data.loc[:,to_be_predicted].values;

# regressor = RandomForestRegressor(n_estimators = 100,n_jobs=100)
# regressor = AdaBoostRegressor(n_estimators = 100,loss='exponential',learning_rate=0.3)
# regressor = ExtraTreesRegressor(n_estimators = 100,n_jobs=100)
regressor = GradientBoostingRegressor(n_estimators = 100,learning_rate=0.5)
regressor = regressor.fit(training_ip, training_op.ravel())
regressor_name = str(regressor).split("(")[0]


MSE_list=[];
R2_list = [];
mean_absolute_error_list = [];
test_start = df.index.searchsorted(datetime(2014, 12, 1,0,1));
days_limit = 60
for step in range(1,days_limit+1):
	print("-------------------------");
	test_end = df.index.searchsorted(datetime(2014, 12, 1,0,1)+timedelta(days=step));
	test_data = df.ix[test_start:test_end];
	test_per = round(float(test_data.Aussentemperatur.count())/float(df.Aussentemperatur.count())*100,2);
	print("testing data count: %d"%(test_data.Aussentemperatur.count()));
	print("testing on %.2f percent data"%(test_per));

	test_ip = test_data.loc[:,to_be_input].values;
	test_actual = test_data.loc[:,to_be_predicted].values;

	test_predictions = regressor.predict(test_ip);

	MSE = mean_squared_error(test_actual, test_predictions)**0.5;
	R2 = r2_score(test_actual, test_predictions);
	mae = mean_absolute_error(test_actual, test_predictions)
	print("Mean Absolute Error : %.2f"%(mae));
	print("Mean Squared Error : %.2f"%(MSE));
	MSE_list.append(MSE);
	R2_list.append(R2);
	mean_absolute_error_list.append(mae);

# print mean_absolute_error_list
pylab.plot(mean_absolute_error_list)
pylab.plot(MSE_list)
# pylab.plot(R2_list)
pylab.legend(["Mean Absolute Error", "RMSE"])

pylab.title(regressor_name + "'s Error of predictions after time delta")
pylab.xlabel("time delta += 1 day")
pylab.ylabel("Error")
ax = pylab.gca()
ax.set_xticks(np.arange(0,days_limit+1,5))
ax.set_yticks(np.arange(0,101,10))
plt.grid()
pylab.savefig('../img/'+regressor_name+'_day_error.png')
# pylab.show()
