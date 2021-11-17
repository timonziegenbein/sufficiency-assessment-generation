import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd


def build_vocab(data_folder, clean_string=True, lower=True):
    vocab = defaultdict(float)

    d = pd.read_csv(data_folder, header=0, delimiter="\t", quoting=3)
    for x in range(len(d)):
        line = d['TEXT'][x]
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev), lower)
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
    return vocab


def build_data2(data_x, data_y, split, clean_string=True, lower=True):
    """
    Loads data and split into 10 folds.
    data_folder[0]: train
    data_folder[1]: dev
    data_folder[2]: test
    """
    instances = []
    for x in range(len(data_x)):
        rev = []
        line = data_x[x]
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev), lower)
        else:
            orig_rev = " ".join(rev).lower()
        label = data_y[x]
        datum = {"y": label, "text": orig_rev, "num_words": len(orig_rev.split()), "split": split}
        instances.append(datum)
    return instances


def build_data(data_folder, split, clean_string=True, lower=True):
    """
    Loads data and split into 10 folds.
    data_folder[0]: train
    data_folder[1]: dev
    data_folder[2]: test
    """
    revs = []

    d = pd.read_csv(data_folder, header=0, delimiter="\t", quoting=3)
    for x in range(len(d)):
        line = d['TEXT'][x]
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev), lower)
        else:
            orig_rev = " ".join(rev).lower()
        label = 0
        if d['ANNOTATION'][x] == 'Sufficiency':
            label = 1
        datum = {"y": label, "text": orig_rev, "num_words": len(orig_rev.split()), "split": split}
        revs.append(datum)

    return revs


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


def clean_str(string, lower=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if lower else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


if __name__=="__main__":    
    w2v_file = "/workspace/ceph_data/input/w2v/GoogleNews-vectors-negative300.bin"

    print "loading data...",
    vocab = build_vocab("data/data-all-tokenized.tsv", clean_string=True, lower=True)
    revs = build_data("data/train.tsv", 0, clean_string=True, lower=True)
    revs += build_data("data/dev.tsv", 1, clean_string=True, lower=True)
    revs += build_data("data/test.tsv", 2, clean_string=True, lower=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of arguments: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("data/suff.p", "wb"))
    print "dataset created!"
