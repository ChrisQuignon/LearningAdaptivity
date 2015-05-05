from imp import load_source
from random import randrange
import numpy as np
import pylab

#import dataset
#equivalent to:
#import ../helpers/csvimport as helper
helper = load_source('dsimport', '../helpers/helper.py')

ds = helper.dsimport()

ds = helper.stretch(ds)

days = helper.chunk_by_month(ds)


#remove dates
days = map(lambda x: helper.split(x, ['Date'])[0], days)


keys = helper.get_all_keys(days[0])

#daymatrix = map (lambda x : helper.as_matrix(x)[0], days)

#create list with dummy zeros for easy appending
values_by_days = []
for _ in range(8):#number of keys
    values_by_days.append([0])

for i, day in enumerate(days):
    day_data, _ = helper.as_matrix(day)
    day_data = day_data.T
    for idx, day_val in enumerate(day_data):
        values_by_days[idx].append([day_val])

#remove dummy zeros
for vals in values_by_days:
    vals.pop(0)

# values_by_days:
# 1st dim: values
# 2nd dim: days
# 3rd dim: values

#substract mean
normset = []
for vals in values_by_days:
    normvals = []
    for s in vals:

        normed = s/ np.linalg.norm(s)
        meansub = normed - np.mean(normed)
        normvals.append(meansub)
    normset.append(normvals)

for i, vals in enumerate(normset):
    # for day in vals:
    pylab.clf()
    pylab.title(keys[i])

    pylab.boxplot(vals, sym='')

    pylab.ylabel('Mean substracted, normalized value')
    month = range(6, 8+len(keys)+1)
    month = map(lambda x: x%12 + 1, month)
    pylab.xticks(range(1, 9), month)
    pylab.xlabel('Month')
    pylab.tight_layout()
    pylab.savefig('../img/boxplot-' + keys[i] + '.png')
    # pylab.show()
