from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import pandas as pd
import data_splitting2
from confusion_matrix import ConfusionMatrix
import cPickle
from keras.utils import np_utils
from callbacks import CaptureBestEpoch
from multiprocessing import Pool
import timeit


def one_fold(iteration):
    it = iteration / 5
    num_fold = iteration % 5
    x = all_folds[iteration / 5][iteration % 5]
    num_fold += 1
    train = x['train']
    test = x['test']
    dev = x['dev']

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

    dev_x = []
    dev_y = []
    for tr in dev:
        dev_x.append(tr['text'])
        dev_y.append(tr['y'])

    # Convert data to vectors
    count_vect = CountVectorizer(max_features=max_words, token_pattern=r"(?u)\b\w+\b", lowercase=True)
    train_x = count_vect.fit_transform(train_x).toarray()
    test_x = count_vect.transform(test_x).toarray()
    dev_x = count_vect.transform(dev_x).toarray()
    # max_words = len(count_vect.get_feature_names())

    # convert labels given in an array (one number per instance) to a matrix including per instance a boolean array
    train_y_cat = np_utils.to_categorical(train_y, nb_labels)
    test_y_cat = np_utils.to_categorical(test_y, nb_labels)
    dev_y_cat = np_utils.to_categorical(dev_y, nb_labels)

    # Define MLP Model:
    model = Sequential()
    model.add(Dense(500, input_dim=max_words, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(125, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop')

    # Capture best model by means of the dev set and store predictions on the test set
    # in order to evaluate the model
    e = CaptureBestEpoch(test_x, test_y_cat)
    model.fit(train_x, train_y_cat, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=False,
              validation_data=(dev_x, dev_y_cat), callbacks=[e])

    # Output confusion matrix on test set
    predictions = e.best_predictions
    confMatrix = ConfusionMatrix(test=test_y, predictions=predictions, header=['noFlaw', 'sufficiency'])
    print "### ITERATION " + str(it)
    print "    Fold " + str(num_fold)
    print "      Accuracy %.3f%%" % (confMatrix.a() * 100)
    print "      Macro F1 %.3f%%" % (confMatrix.macro_f_impl1() * 100)
    accuracy = confMatrix.a()
    f1 = confMatrix.macro_f_impl1()
    result = {'iteration': i, 'fold': num_fold, 'model': 'BaselineHeuristic', 'acc': confMatrix.a(),
                    'f1': confMatrix.macro_f_impl1(), 'predictions:': predictions, 'confusion_matrix': confMatrix,
                    'best_epoch': e.best_epoch}

    return i, result, accuracy, f1

# timer (DEBUG)
start = timeit.default_timer()

random.seed(123)
np.random.seed(123)

nb_epoch = 30
batch_size = 100
max_words = 4000
nb_labels = 2

# read data
path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/DATA-TSV/data-all-tokenized.tsv'
data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)

cores = 4
k = 5
iterations = 20
sum_accuracy = 0.0
sum_f1 = 0.0
results = []
resultPath = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency2/Results-MLP/report_MLP1.bin'

all_folds = []
for i in range(iterations):
    all_folds.append(data_splitting2.get_cv_data(data, k))

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