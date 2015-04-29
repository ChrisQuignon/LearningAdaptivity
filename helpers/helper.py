#!/usr/bin/env python

import csv
import numpy as np
from datetime import datetime, timedelta

import sys


def dsimport():
    #target data structure
    ds = []

    #IMPORT FILE
    input_file = csv.DictReader(open("../data/ds1_weather.csv"), delimiter=';',)
    for row in input_file:

        d = {}

        for key in row.keys():
            row[key] = row[key].replace(',', '.') #german number conversion
            #Data conersion
            if row[key]:
                #date
                if key == 'Date':
                    #DD.MM.YYYY HH:MM:SS
                    # date, time = row['Date'].split(' ')
                    #
                    # DD, MM, YYYY = map(int, date.split('.'))
                    #
                    # hh, mm, ss = map(int, time.split(':'))
                    # d[key] = datetime(year=YYYY,
                    #                 month=MM,
                    #                 day=DD,
                    #                 hour=hh
                    #                 )
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
    return ds

def stretch(ds):
    r = []
    #find all first values
    keys = get_all_keys(ds)

    firstvals={}
    for key in keys:
        firstvals[key] = ''

    for d in ds:
        for key in keys:
            if key in d.keys():
                firstvals[key]=d[key]
        if not any(map(lambda x: firstvals[x] == '', keys)): #we still have an emppty first val
            break

    for d in ds:
        for key in keys:
            if key in d.keys():
                firstvals[key] = d[key]
            else:
                d[key] = firstvals[key]

    return ds




def get_all_keys(ds):
    keys = []
    for d in ds:
         for k in d.keys():
             if not( k in keys):
                 keys.append(k)
    return keys

def as_matrix(ds):

    keys = get_all_keys(ds)

    m = []
    for d in ds:
        row = [d[key] for key in keys]
        m.append(row)
    npm = np.asmatrix(m)

    return npm, keys

def extract_keys(ds, keys):
    ds = ds
    keys = get_all_keys(ds)

    extracts = []
    rest = ds

    for i in range(len(keys)):
        remove_idx = keys.index(keys[i])

        extract = np.asarray(rest[:,training_idx])
        rest = np.delete(rest, (training_idx), axis = 1)

        extracts.append(extract)
    return extracts, rest

#split into input and output
def split(ds, output_keys):
    outputs = []
    inputs = []

    for d in ds:
        output = {}
        input = {}
        for key in d.keys():
            if key in output_keys:
                output[key] = d[key]
            #we exclude the key
            elif key == 'Date':
                pass
            else:
                input[key] = d[key]
        inputs.append(input)
        outputs.append(output)

    return inputs, outputs


#Do your stuff

# #HELPERS


# #houry data:
# [d for d in ds if d['Date'].hour == 0]

# #minutely data:
# [d for d in ds if d['Date'].minute == 0]

# #only one key:
# [d['Leistung'] for d in ds if 'Leistung' in d.keys()]

# #all data between two dates
# min_d = datetime.strptime('01.07.2014 00:01:14', '%d.%m.%Y %H:%M:%S')
# max_d = datetime.strptime('12.07.2014 00:01:14', '%d.%m.%Y %H:%M:%S')
# [d for d in ds if min_d < d['Date'] < max_d]

# #next one hour of data
# now = ds[0]['Date']
# [d for d in ds if d['Date']-now < timedelta(hours=1)]

# #sort by date
# ds = sorted(ds, key = lambda x : x['Date'])

# #min of key
# min([d for d in ds if 'Leistung' in d.keys()], key= lambda x: x['Leistung'])

# #chunk
# from itertools import groupby
# for year_k, year in groupby(ds, key = lambda x : x['Date'].year):
#     #year = dataset per year
#     for month_k, month in groupby(year, key = lambda x: x['Date'].month):
#         #month = dataset per month
#         for day_k, day in groupby(month, key = lambda x: x ['Date'].day):
#             #day = dataset per day
#             print day_k
