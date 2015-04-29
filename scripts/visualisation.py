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
                date, time = row['Date'].split(' ')

                DD, MM, YYYY = map(int, date.split('.'))

                hh, mm, ss = map(int, time.split(':'))
                d[key] = datetime(year=YYYY,
                                month=MM,
                                day=DD,
                                hour=hh
                                )
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

#Do your stuff
