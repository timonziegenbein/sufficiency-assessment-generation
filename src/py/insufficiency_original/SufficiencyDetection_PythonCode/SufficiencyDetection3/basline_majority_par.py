import numpy as np
import random
import pandas as pd
import data_splitting
from confusion_matrix import ConfusionMatrix
import cPickle
from multiprocessing import Pool
import timeit


def get_predictions(d):
    return [0 for x in range(len(d))]


def one_fold(i):
    it = i/5
    num_fold = i % 5
    x = all_folds[i/5][i % 5]
    train = x['train']
    test = x['test']

    train_x = []
    train_y = []
    for tr in train:
        train_x.append(tr['text'])
        train_y.append(tr['y'])

    test_x = []
    test_y = []
    for tr in test:
        test_x.append(tr['text'])
        test_y.append(tr['y'])

    # Get outcome of heuristic baseline
    predictions = get_predictions(test_x)
    conf_matrix = ConfusionMatrix(test=test_y, predictions=predictions, header=['noFlaw', 'sufficiency'])

    print "### ITERATION " + str(it)
    print "    Fold " + str(num_fold)
    print "      Accuracy %.3f%%" % (conf_matrix.a() * 100)
    print "      Macro F1 %.3f%%" % (conf_matrix.macro_f_impl1() * 100)
    accuracy = conf_matrix.a()
    f1 = conf_matrix.macro_f_impl1()
    result = {'iteration': it, 'fold': num_fold, 'model': 'BaselineMajority', 'acc': conf_matrix.a(),
              'f1': conf_matrix.macro_f_impl1(), 'predictions:': predictions, 'confusion_matrix': conf_matrix}
    return i, result, accuracy, f1

# timer (DEBUG)
start = timeit.default_timer()

random.seed(123)
np.random.seed(123)

# read data
path = '/workspace/ceph_data/input/UKP-InsufficientArguments_v1.0/data-tokenized.tsv'
data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)

cores = 4
k = 5
iterations = 20
sum_accuracy = 0.0
sum_f1 = 0.0
results = []
resultPath = '/workspace/ceph_data/output/insufficiency_original/report_par.bin'

all_folds = []
for i in range(iterations):
    all_folds.append(data_splitting.get_stratified_cv_data(data, k))

# if required for windows
if __name__ == '__main__':
    # initialization parallel processes
    pool = Pool(processes=cores)
    # run parallel processes and resolve results

    pool_returns = zip(*(pool.map(one_fold, range(iterations*k))))

    results = pool_returns[1]
    sum_accuracy = sum(pool_returns[2])
    sum_f1 = sum(pool_returns[3])

    print "\nAvg-Accuracy %.3f%%" % (sum_accuracy / (iterations * k) * 100)
    print "Avg-Macro-F1 %.3f%%" % (sum_f1 / (iterations * k) * 100)

    # save result report
    f = file(resultPath, 'wb')
    cPickle.dump(results, f)
    f.close()

    # timer (DEBUG)
    stop = timeit.default_timer()

    print "runtime: ", stop - start
