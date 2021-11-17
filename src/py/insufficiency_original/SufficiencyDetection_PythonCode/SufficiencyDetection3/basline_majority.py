import numpy as np
import random
import pandas as pd
import data_splitting2
from confusion_matrix import ConfusionMatrix
import cPickle


def get_predictions(d):
    return [0 for x in range(len(d))]

random.seed(123)
np.random.seed(123)

# read data
path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/DATA-TSV/data-all-tokenized.tsv'
data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)

k = 5
iterations = 20
sum_accuracy = 0.0
sum_f1 = 0.0
results = []
resultPath = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Baseline-majority/baseline-majority.bin'


for i in range(iterations):
    print "### ITERATION " + str(i)
    folds = data_splitting2.get_cv_data(data, k)
    num_fold = 0
    for x in folds:
        num_fold += 1
        print "    Fold " + str(num_fold)
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
        confMatrix = ConfusionMatrix(test=test_y, predictions=predictions, header=['noFlaw', 'sufficiency'])

        print "      Accuracy %.3f%%" % (confMatrix.a() * 100)
        print "      Macro F1 %.3f%%" % (confMatrix.macro_f_impl1() * 100)
        sum_accuracy += confMatrix.a()
        sum_f1 += confMatrix.macro_f_impl1()
        results.append({'iteration': i, 'fold': num_fold, 'model': 'BaselineMajority', 'acc': confMatrix.a(),
                        'f1': confMatrix.macro_f_impl1(), 'predictions:': predictions, 'confusion_matrix': confMatrix})

print "\nAvg-Accuracy %.3f%%" % (sum_accuracy / (iterations * k) * 100)
print "Avg-Macro-F1 %.3f%%" % (sum_f1 / (iterations * k) * 100)

# save result report
f = file(resultPath, 'wb')
cPickle.dump(results, f)
f.close()
