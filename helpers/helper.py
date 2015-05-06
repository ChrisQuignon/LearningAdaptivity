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
            row[key] = row[key].replace(',', '.') #english number conversion
            #Data conversion
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
    """Projects all values into datasets where the value is notpresent"""
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
    """Returns all keys in a dataset"""
    keys = []
    for d in ds:
         for k in d.keys():
             if not( k in keys):
                 keys.append(k)
    return keys

def as_matrix(ds):
    """Returns the dataset as a matrix and the corresponding keys mapping the dataset to the matrix"""

    keys = get_all_keys(ds)

    m = []
    for d in ds:
        row = [d[key] for key in keys]
        m.append(row)
    npm = np.asarray(m)

    return npm, keys

# not in use
# def extract_keys(ds, keys):
#     ds = ds
#     keys = get_all_keys(ds)
#
#     extracts = []
#     rest = ds
#
#     for i in range(len(keys)):
#         remove_idx = keys.index(keys[i])
#         print remove_idx
#
#         extract = np.asarray(rest[:,remove_idx])
#         rest = np.delete(rest, (remove_idx), axis = 1)
#
#         extracts.append(extract)
#     return extracts, rest

#split into input and output
def split(ds, output_keys):
    """Splits off the keys from the given dataset"""
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

# #HELPERS
def get_values_by_key(ds, key):
    """Returns all values with the given key"""
    return [d[key] for d in ds if key in d.keys()]


def in_timerange(ds, start, end):
    """Returns all data within a given timeframe.s
    Example calls:
    in_timerange(ds, '01.07.2014 00:01:14', '12.07.2014 00:01:14')
    in_timerange(ds, ds[0], ds[100])
    in_timerange(ds, ds[0]['Date'], ds[100]['Date'])
    """
    if isinstance(start, str) and  isinstance(end, str):
        start = datetime.strptime(start, '%d.%m.%Y %H:%M:%S')
        end = datetime.strptime(end, '%d.%m.%Y %H:%M:%S')

    if isinstance(start, dict) and isinstance(end, dict):
        min_d = min(start['Date'], end['Date'])
        max_d = max(start['Date'], end['Date'])

    if isinstance(start, datetime) and isinstance(end, datetime):
        min_d = min(start, end)
        max_d = max(start, end)

    return [d for d in ds if min_d < d['Date'] < max_d]


def in_timedelta(ds, start, timedelta):
    """Returns all data that is withing timedelta after the given delta
    Example call: in_timedelta(ds, ds[0], timedelta(hours = 1))"""
    start = start['Date']
    return [d for d in ds if d['Date']-start < timedelta]

def sort_by_key(ds, key):
    """Returns the dataset sorted by a given key.
    example call: sort_by_key(ds, 'Dates')"""
    return sorted(ds, key = lambda x : x[key])


def min_of_key(ds, key):
    """Returns the data with the minimal value in the Keyset
    Example call: min_of_keys(ds, 'Leistung')"""
    return min([d for d in ds if key in d.keys()], key= lambda x: x[key])


def max_of_key(ds, key):
    """Returns the data with the maxsimal value in the Keyset
    Example call: max_of_keys(ds, 'Leistung')"""
    return max([d for d in ds if key in d.keys()], key= lambda x: x[key])

#CHUNKS

def chunk_by_month(ds):
    """Returns a list of datasets where all values in a list are on the same month
    Example call: chunk_by_days(ds)"""
    from itertools import groupby

    chunks = []
    for year_k, year in groupby(ds, key = lambda x : x['Date'].year):
        #year = dataset per year
        for month_k, month in groupby(year, key = lambda x: x['Date'].month):
            #month = dataset per month
            chunks.append([d for d in month])
    return chunks


def chunk_by_days(ds):
    """Returns a list of datasets where all values in a list are on the same day
    Example call: chunk_by_days(ds)"""
    from itertools import groupby

    chunks = []
    for year_k, year in groupby(ds, key = lambda x : x['Date'].year):
        #year = dataset per year
        for month_k, month in groupby(year, key = lambda x: x['Date'].month):
            #month = dataset per month
            for day_k, day in groupby(month, key = lambda x: x ['Date'].day):
                #day = dataset per day
                chunks.append([d for d in day])
    return chunks


def chunk_by_hours(ds):
    """Returns a list of datasets where all values in a list are in the same hour
    Example call: chunk_by_hour(ds)"""
    from itertools import groupby

    chunks = []
    for year_k, year in groupby(ds, key = lambda x : x['Date'].year):
        #year = dataset per year
        for month_k, month in groupby(year, key = lambda x: x['Date'].month):
            #month = dataset per month
            for day_k, day in groupby(month, key = lambda x: x ['Date'].day):
                #day = dataset per day
                for hour_k, hour in groupby(month, key = lambda x: x ['Date'].hour):
                    #hour = dataset per hour
                    chunks.append([d for d in hour])
    return chunks



def chunk_by_n_minutes(ds, n):
    """Returns a a list of datasets where all values in a list are on the same n minutes
    N has to be a fraction of 60
    Example call: chunk_by_n_minutes(ds, 5)"""
    from itertools import groupby

    if not (60%n == 0):
        print "ERROR, n (=", n, ") can only be a fraction of 60"
        return []

    chunks = []
    for year_k, year in groupby(ds, key = lambda x : x['Date'].year):
        #year = dataset per year
        for month_k, month in groupby(year, key = lambda x: x['Date'].month):
            #month = dataset per month
            for day_k, day in groupby(month, key = lambda x: x ['Date'].day):
                #day = dataset per day
                for hour_k, hour in groupby(month, key = lambda x: x ['Date'].hour):
                    #hour = dataset per hour
                    for minute_k, minutes in groupby(month, key = lambda x: x ['Date'].minute/n):
                        #minutes = dataset per n minutes
                        chunks.append([d for d in minutes])
    return chunks

#subsampling


def subsample_hours(ds):
    """Subsamples a given datatset ds with a mean per hour"""
    samples = chunk_by_hours(ds)
    keys = get_all_keys(samples[0])

    s = []
    for sample in samples:
        d = {}
        for key in keys:
            if key == 'Date':
                d[key] = min(get_values_by_key(sample, key))
            else:
                d[key] = np.mean(get_values_by_key(sample, key))
        s.append(d)
    return s


def subsample_days(ds):
    """Subsamples a given datatset ds with a mean per day"""
    samples = chunk_by_days(ds)
    keys = get_all_keys(samples[0])

    s = []
    for sample in samples:
        d = {}
        for key in keys:
            if key == 'Date':
                d[key] = min(get_values_by_key(sample, key))
            else:
                d[key] = np.mean(get_values_by_key(sample, key))
        s.append(d)
    return s


def subsample_n_minutes(ds, n):
    """Subsamples a given datatset ds with a mean per hour"""
    samples = chunk_by_n_minutes(ds, n)
    keys = get_all_keys(samples[0])

    s = []
    for sample in samples:
        d = {}
        for key in keys:
            if key == 'Date':
                d[key] = min(get_values_by_key(sample, key))
            else:
                d[key] = np.mean(get_values_by_key(sample, key))
        s.append(d)
    return s


def translate(word):
    """Translates the given german feature name into english.
    Also works for lists of german feature names"""
    if isinstance(word, list):
        return map(translate, word)
    if isinstance(word, np.ndarray):
        return translate(word.tolist())

    dict = {
    'Vorlauftemperatur':'input temperature',
    'Volumenstrom':'volumetric flowrate',
    'Ruecklauftemperatur':'output temperature',
    'Aussentemperatur':'outside temperature',
    'Leistung':'power',
    'Niederschlag' :'precipitation',
    'Energie':'energy',
    'Date':'date',
    'Relative Feuchte':'relative air humidity'
    }

    if word in dict.keys():
        return dict[word]
    else:
        print word, ' unkown!'
        return
