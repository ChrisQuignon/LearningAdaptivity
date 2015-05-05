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

months = helper.chunk_by_month(ds)


#remove dates
months = map(lambda x: helper.split(x, ['Date'])[0], months)


keys = helper.get_all_keys(months[0])

#monthmatrix = map (lambda x : helper.as_matrix(x)[0], months)

#create list with dummy zeros for easy appending
values_by_months = []
for _ in range(8):#number of keys
    values_by_months.append([0])

for i, month in enumerate(months):
    month_data, _ = helper.as_matrix(month)
    month_data = month_data.T
    for idx, month_val in enumerate(month_data):
        values_by_months[idx].append([month_val])

#remove dummy zeros
for vals in values_by_months:
    vals.pop(0)

# values_by_months:
# 1st dim: features
# 2nd dim: months
# 3rd dim: values

#substract mean
normset = []
for vals in values_by_months:
    normvals = []
    for s in vals:

        normed = s/ np.linalg.norm(s)
        meansub = normed - np.mean(normed)
        normvals.append(meansub)
    normset.append(normvals)

for i, vals in enumerate(normset):
    # for month in vals:
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
