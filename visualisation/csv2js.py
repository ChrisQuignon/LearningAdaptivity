#!/usr/bin/env python

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from datetime import datetime

import sys



#target data structure
ds = []

#IMPORT FILE
input_file = csv.DictReader(open("data/ds1_weather.csv"), delimiter=';',)
for row in input_file:

    d = {}

    for key in row.keys():
        row[key] = row[key].replace(',', '.') #german number conversion
        #Data conersion
        if row[key]:
            #date
            if key == 'Date':
                #DD.MM.YYYY HH:MM:SS
                d[key] = datetime.strptime(row[key], '%d.%m.%Y %H:%M:%S')

            #sensors: minute
            elif key ==  'Leistung':
                d[key] =float(row['Leistung'])

            elif key == 'Ruecklauftemperatur':
                d[key] = float(row['Ruecklauftemperatur'])

            elif key == 'Vorlauftemperatur':
                d[key] = float(row['Vorlauftemperatur'])

            elif key == 'Volumenstrom':
                d[key] = float(row['Volumenstrom'])

            #energy:dail
            elif key == 'Energie':
                d[key] = float(row['Energie'])

            #weather: hour
            elif key == 'Aussentemperatur':
                d[key] = float(row['Aussentemperatur'])

            elif key == 'Relative Feuchte':
                d[key] = int(row['Relative Feuchte'])

            elif key == 'Niederschlag':
                d[key] = float(row['Niederschlag'])

            else:
                print 'ERROR: Key ' + key + 'not known'

    ds.append(d)
# print len(ds)

#GENERATE HIGHCHARTS .js file

listedd = {}

for d in ds:
    for key in filter(lambda x : 'Date' not in x, d.keys()):
        if not key in listedd.keys():
            listedd[key] = [[d['Date'], d[key]]]
        else:
            listedd[key].append([d['Date'], d[key]])

for key in listedd:
    sys.stdout.write("var " + "".join(key.split()) +"=")
    sys.stdout.write("[")

    for idx, entry in enumerate(listedd[key]):
        date, val = entry
        if idx == 0:
            sys.stdout.write('[' + date.strftime("%s000") + ', ' + str(val) + ']')
        else:
            sys.stdout.write(',\n [' + date.strftime("%s000") + ', ' + str(val) + ']')
    sys.stdout.write('];\n')


#
# l = []
# s = "{\n \n name: '" + key +"',\n data: "
