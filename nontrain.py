#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from sklearn import svm
# from nltk.corpus import wordnet as wn
from nltk import pos_tag


def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def string_match(h, ref, n):
    h_ngram = []
    ref_ngram = []
    for i in range(len(h) - n + 1):
        w = ""
        for j in range(n):
             w += h[i + j]
        h_ngram.append(w)
    
    for i in range(len(ref) - n + 1):
        w = ""
        for j in range(n):
             w += ref[i + j]
        ref_ngram.append(w)
    
    rset = set(ref_ngram)
    h_match = word_matches(h_ngram, rset)
    
    p = h_match * 1.0 / len(rset)
    r = h_match * 1.0 / len(h_ngram)
    f = 0.0
    if p + r != 0:
        f = p * r * 2.0 / (p + r)
    return f

def pos_match(h_pos, ref_pos):
    h = []
    ref = []
    for pos in h_pos:
        h.append(pos[1])
    for pos in ref_pos:
        ref.append(pos[1])
    
    return n_gram(h, ref, 1)

def n_gram(h, ref, start):
    ans = 0
    for i in range(start, 5):
        if i > len(h) or i > len(ref):
            break
        ans += string_match(h, ref, i)
    return ans

def simple_meteor(h, ref):
    alpha = 0.8
    rset = set(ref)
    h_match = word_matches(h, rset)
    p = (h_match * 1.0) / len(rset)
    r = (h_match * 1.0) / len(h)
    l = 0
    if p + r != 0:
        l = p * r/ ((1 - alpha) * p + alpha * r)
    return l
def feature_evalutaion(h, ref):
    l = 0
    # String match
    l += n_gram(h, ref, 2)
    # POS
    h_pos = pos_tag(h)
    ref_pos = pos_tag(ref)
    l += pos_match(h_pos, ref_pos)
    
    return l
def main():
    from sklearn import svm
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/hyp1-hyp2-ref',
            help='input file (default data/hyp1-hyp2-ref)')
    parser.add_argument('-g', '--golden', default='data/dev.answers',
            help='input file (default data/dev.answers)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
    

    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

    print "Preparing training data..."
    TRAIN_SIZE = 10000
    
    dev_train = []
    dev_class = []
     
    ln = 0
    with open(opts.golden) as f:  
        for line in f:
            if ln <  TRAIN_SIZE:
                cl = int(line)
                dev_class.append(cl)          
                ln += 1
            else:
                break

   
    ln = 0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        if ln < TRAIN_SIZE:
            l1 = feature_evalutaion(h1, ref) + simple_meteor(h1, ref)
            l2 = feature_evalutaion(h2, ref) + simple_meteor(h2, ref)
            simi = feature_evalutaion(h1, h2) + simple_meteor(h1, h2)
            fts = [l1, l2, simi]
            dev_train.append(fts)
            ln += 1
        else:
            break
    
    print "Finish preparation"
    print len(dev_class)
    print len(dev_train)
    
    print "Start to train..."
    clf = svm.SVC(kernel='linear')
    clf.fit(dev_train, dev_class)
    print "Training is finished"
    
    # predict
    f = open("eval.out", 'w')
    print "Predicting on train..."
    
    idx = 0
    right = 0
    for fts in dev_train:
        ans = clf.predict([fts])
        if dev_class[idx] == ans:
            right += 1
        idx += 1
        pred = ans[0]
        if fts[0] > fts[1] and pred == 1:
            f.write("1\n")
        elif fts[0] < fts[1] and pred == -1:
            f.write("-1\n")
        else:
            f.write("0\n")
   
    print right
if __name__ == '__main__':
    main()