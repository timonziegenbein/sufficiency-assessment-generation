import numpy as np
import random
import pandas as pd
import data_splitting2
from confusion_matrix import ConfusionMatrix
import cPickle
from process_data import build_vocab, build_data2, load_bin_vec, add_unknown_words, get_W
from conv_net import make_idx_data, train_conv_net
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

    revs = build_data2(train_x, train_y, 0, clean_string=True, lower=True)
    revs += build_data2(dev_x, dev_y, 1, clean_string=True, lower=True)
    revs += build_data2(test_x, test_y, 2, clean_string=True, lower=True)

    max_l = np.max(pd.DataFrame(revs)["num_words"])

    data_sets = make_idx_data(revs, word_idx_map, max_l=max_l, k=300, filter_h=5)

    # print "Train:" + str(len(data_sets[0]))
    # print "Dev  :" + str(len(data_sets[1]))
    # print "Test :" + str(len(data_sets[2]))

    model_results = train_conv_net(data_sets, U, lr_decay=0.95, filter_hs=filter_hs, conv_non_linear="tanh",
                                    hidden_units=[250, 2], shuffle_batch=True, n_epochs=nb_epoch, sqr_norm_lim=9,
                                    non_static=non_static, batch_size=batch_size, dropout_rate=[0.5])

    # Output confusion matrix on test set
    predictions = model_results['pred']
    best_epoch = model_results['best_epoch']
    # print len(predictions)
    # print len(test_y)
    conf_matrix = ConfusionMatrix(test=test_y, predictions=predictions, header=['noFlaw', 'sufficiency'])
    print "### ITERATION " + str(it)
    print "    Fold " + str(num_fold)
    print "      Accuracy %.3f%%" % (conf_matrix.a() * 100)
    print "      Macro F1 %.3f%%" % (conf_matrix.macro_f_impl1() * 100)
    accuracy = conf_matrix.a()
    f1 = conf_matrix.macro_f_impl1()
    result = {'iteration': it, 'fold': num_fold, 'model': 'BaselineHeuristic', 'acc': conf_matrix.a(),
            'f1': conf_matrix.macro_f_impl1(), 'predictions:': predictions, 'confusion_matrix': conf_matrix,
            'best_epoch': best_epoch}
    return i, result, accuracy, f1

# timer (DEBUG)
start = timeit.default_timer()

random.seed(123)
np.random.seed(123)

nb_epoch = 30
batch_size = 50
filter_hs = [2]
non_static = True  # static = False; non-static = True
word_vectors = "-word2vec"  # -rand

# read data
path = '/workspace/ceph_data/intermediate/insufficiency-original-sub-pc/data-tokenized.tsv'
w2v_file = "/workspace/ceph_data/input/w2v/GoogleNews-vectors-negative300.bin"
data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)

cores = 5
k = 5
iterations = 2
sum_accuracy = 0.0
sum_f1 = 0.0
results = []
resultPath = '/workspace/ceph_data/output/insufficiency-original-sub-pc/report_CNN_pc.bin'


all_folds = []
for i in range(iterations):
    all_folds.append(data_splitting2.get_cv_data(data, k))

# get vocabulary and read word2vec
vocab = build_vocab(path, clean_string=True, lower=True)
print "vocab size: " + str(len(vocab))
w2v = load_bin_vec(w2v_file, vocab)
print "word2vec loaded!"
add_unknown_words(w2v, vocab)
W, word_idx_map = get_W(w2v)
rand_vecs = {}
add_unknown_words(rand_vecs, vocab)
W2, _ = get_W(rand_vecs)
if word_vectors == "-rand":
    print "using: random vectors"
    U = W2
elif word_vectors == "-word2vec":
    print "using: word2vec vectors"
    U = W

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
