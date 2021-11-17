import numpy as np
import random
import pandas as pd


def split_labels(data):
    pos = []
    neg = []
    for x in range(len(data)):
        label = 0
        if data['ANNOTATION'][x] == 'insufficient':
            label = 1
        text = data['TEXT'][x]
        essay_id = data['ESSAY'][x]
        arg = data['ARGUMENT'][x]
        datum = {"id": essay_id, "arg": arg, "y": label, "text": text, "num_words": len(text.split())}

        if label == 1:
            pos.append(datum)
        else:
            neg.append(datum)
    return pos, neg


def n_chunks(l, n):
    """ Yield n successive chunks from l.
    """
    new = int(1.0 * len(l) / n + 0.5)
    for i in xrange(0, n - 1):
        yield l[i * new:i * new + new]
    yield l[n * new - new:]


def split_train_test_dev_sets(k, dev_prop, pos, neg):
    d = []
    for a in range(k):
        test = pos[a] + neg[a]
        pos_tmp = []
        neg_tmp = []
        for b in range(k):
            if b != a:
                pos_tmp = pos_tmp + pos[b]
                neg_tmp = neg_tmp + neg[b]
        random.shuffle(pos_tmp)
        random.shuffle(neg_tmp)
        train = pos_tmp[:int(np.round(len(pos_tmp)*(1-dev_prop)))]+neg_tmp[:int(np.round(len(neg_tmp) * (1-dev_prop)))]
        dev = pos_tmp[int(np.round(len(pos_tmp)*(1-dev_prop))):]+neg_tmp[int(np.round(len(neg_tmp) * (1-dev_prop))):]
        d.append({'test': test, 'train': train, 'dev': dev})
    return d


def get_stratified_cv_data(data, k=5):
    # stratification
    pos, neg = split_labels(data)

    # shuffle
    random.shuffle(pos)
    random.shuffle(neg)

    # buckets for CV
    pos_buckets = list(n_chunks(pos, k))
    neg_buckets = list(n_chunks(neg, k))

    # split in train, test, dev
    folds = split_train_test_dev_sets(k, 0.1, pos_buckets, neg_buckets)
    return folds


def test_folds(fold_data):
    all_data = []
    for f in fold_data:
        test = f['test']
        dev = f['dev']
        train = f['train']
        l = len(test) + len(train) + len(dev)

        for t in test:
            if not t in all_data:
                all_data.append(t)
            else:
                print "ERROR: Instance included in several test sets"
        print "SumFold: " + str(l)
        print "  train" + str(len(train))
        print_distribution(train)
        print "  test" + str(len(test))
        print_distribution(test)
        print "  dev" + str(len(dev))
        print_distribution(dev)
    print "SumAll : " + str(len(all_data))


def print_distribution(dist):
    positive = 0
    negative = 0
    for v in dist:
        if v['y'] == 1:
            positive += 1
        else:
            negative += 1
    print '    positive:\t%.3f%%' % (float(positive)/float(len(dist)))
    print '    negative:\t%.3f%%' % (float(negative)/float(len(dist)))


# random.seed(123)
# np.random.seed(123)
#
# # read data
path = '/workspace/ceph_data/input/UKP-InsufficientArguments_v1.0/data-tokenized.tsv'
data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)

k = 5
# iterations = 20
# dict = {}
# num_docs = 0
# for i in range(iterations):
#     folds = get_stratified_cv_data(data, k)
#     for f in folds:
#         for e in f['test']:
#             if not 'essay' + str(e['id']).zfill(3) + '_' + str(e['arg']) in dict:
#                 dict['essay' + str(e['id']).zfill(3) + '_' + str(e['arg'])] = []
#
#             dict['essay' + str(e['id']).zfill(3) + '_' + str(e['arg'])] += ['TEST']
#         for e in f['dev']:
#             if not 'essay' + str(e['id']).zfill(3) + '_' + str(e['arg']) in dict:
#                 dict['essay' + str(e['id']).zfill(3) + '_' + str(e['arg'])] = []
#             dict['essay' + str(e['id']).zfill(3) + '_' + str(e['arg'])] += ['DEV']
#         for e in f['train']:
#             if not 'essay' + str(e['id']).zfill(3) + '_' + str(e['arg']) in dict:
#                 dict['essay' + str(e['id']).zfill(3) + '_' + str(e['arg'])] = []
#             dict['essay' + str(e['id']).zfill(3) + '_' + str(e['arg'])] += ['TRAIN']
#
# f = open('folds2.tsv', 'w')
# for arg in sorted(dict):
#     line = ""
#     line += arg + "\t"
#     for set in dict[arg]:
#         line += set + "\t"
#     f.write(line + "\n")
# f.close()
