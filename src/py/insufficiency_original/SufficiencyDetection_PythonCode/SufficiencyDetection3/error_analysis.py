import pandas as pd
import random
import numpy as np
import data_splitting
import cPickle


def get_doc_id(iteration, fold, index):
    x = all_folds[iteration][fold]
    test = x['test']
    return str(test[index]['id']) + '_' + str(test[index]['arg'])


random.seed(123)
np.random.seed(123)

k = 5
iterations = 20

# read data
path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency/DATA-TSV/data-all-tokenized.tsv'
data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)

# get folds
all_folds = []
for i in range(iterations):
    all_folds.append(data_splitting.get_stratified_cv_data(data, k))

# load report file
resultPath = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency/Results-CNN/report_CNN-filter_2_par.bin'
f = file(resultPath, 'rb')
report = cPickle.load(f)
f.close()
for e in report:
    i = e['iteration']
    fold = e['fold']-1
    print 'Experiment: iteration=' + str(i) + '; fold=' + str(fold)
    pred = e['predictions:']
    for index in range(len(pred)):
        p = pred[index]
        label = False
        if p:
            label = True
        print "   " + get_doc_id(i, fold, index) + "  " + str(label)

