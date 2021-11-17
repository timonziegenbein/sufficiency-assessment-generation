import numpy as np
import random
import pandas as pd


def format_data(data):
    d = []
    for x in range(len(data)):
        label = 0
        if data['ANNOTATION'][x] == 'insufficient':
            label = 1
        text = data['TEXT'][x]
        essay_id = data['ESSAY'][x]
        arg = data['ARGUMENT'][x]
        datum = {"id": essay_id, "arg": arg, "y": label, "text": text, "num_words": len(text.split())}

        if label == 1:
            d.append(datum)
        else:
            d.append(datum)
    return d


def combine_arguments(data):
    ca = {}
    for x in data:
        current_id = x["id"]
        if not current_id in ca:
            ca[current_id] = {"id": x["id"], "args": [], "pos": 0, "neg": 0}
        item = ca[current_id]
        item["args"].append(x)
        if x["y"] == 1:
            item["pos"] += 1
        else:
            item["neg"] += 1
    l = []
    for x in ca:
        l.append(ca[x])
    return l


def decompose_arguments(data):
    l = []
    for x in data:
        for y in x["args"]:
            l.append(y)
    return l


def split_labels(data):
    pos = []
    neg = []
    for x in data:
        if x["y"] == 1:
            pos.append(x)
        else:
            neg.append(x)
    return pos, neg


def n_chunks(l, n):
    """ Yield n successive chunks from l.
    """
    new = int(1.0 * len(l) / n + 0.5)
    for i in xrange(0, n - 1):
        yield l[i * new:i * new + new]
    yield l[n * new - new:]


def split_train_test_dev_sets(k, dev_prop, buckets):
    d = []
    for a in range(k):
        test = buckets[a]
        tmp = []
        for b in range(k):
            if b != a:
                tmp = tmp + buckets[b]
        random.shuffle(tmp)
        train = tmp[:int(np.round(len(tmp)*(1-dev_prop)))]
        dev = tmp[int(np.round(len(tmp)*(1-dev_prop))):]

        d.append({'test': decompose_arguments(test), 'train': decompose_arguments(train), 'dev': decompose_arguments(dev)})
    return d


def get_cv_data(data, k=5):

    # format data
    d = format_data(data)

    d = combine_arguments(d)

    random.shuffle(d)

#   print len(d)
#   for i in range(len(d)):
#       print str(d[i]["args"][0]["id"]) + "  pos:" + str(d[i]["pos"]) + "  neg:" + str(d[i]["neg"]) + "  all:" + str(len(d[i]["args"]))

    buckets = list(n_chunks(d, k))

    # split in train, test, dev
    folds = split_train_test_dev_sets(k, 0.1, buckets)
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
        print "  train " + str(len(train))
        print_distribution(train)
        print "  test " + str(len(test))
        print_distribution(test)
        print "  dev " + str(len(dev))
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
    print '    positive (Insufficient):\t%.3f%%' % (float(positive)/float(len(dist)))
    print '    negative (Sufficient):\t%.3f%%' % (float(negative)/float(len(dist)))


# random.seed(123)
# np.random.seed(123)
#
# # read data
# path = '/Users/zemes/DEVELOPMENT/Experiments/Sufficiency/DATA-TSV/data-all-tokenized2.tsv'
# data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)
#
# k = 5
# iterations = 20
# dict = {}
# num_docs = 0
# for i in range(iterations):
#     folds = get_cv_data(data, k)
#     test_folds(folds)
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
#
# f = open('folds2.tsv', 'w')
# for arg in sorted(dict):
#     line = ""
#     line += arg + "\t"
#     for set in dict[arg]:
#         line += set + "\t"
#     f.write(line + "\n")
# f.close()
