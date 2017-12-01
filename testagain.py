#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from sklearn import svm
# from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus import wordnet as wn

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
    
    return n_gram(h, ref)

def n_gram(h, ref):
    ans = 0
    for i in range(1, 5):
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
def penalty(h, ref):
    hset = set(h)
    count = 0
    l = []
    for i in range(len(ref)):
        if ref[i] in hset:
            count += 1
        elif count > 0:
            l.append(count)
            count = 0
    if len(l) > 0:
        chunks = max(l)
        match = sum(l)
        return 0.5 * chunks / match
    return 0

def feature_evalutaion(h, ref):
    l = 0
    # String match
    l = n_gram(h, ref)
    # POS
    h_pos = pos_tag(h)
    ref_pos = pos_tag(ref)
    p = pos_match(h_pos, ref_pos)
    
    return l, p
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
    linenum = set([])
    ln = 0
    zeros = 0
    with open(opts.golden) as f:  
        for line in f:
            if len(dev_class) < TRAIN_SIZE:
                cl = int(line)
                if cl != 0:
                    if len(dev_class) - zeros < TRAIN_SIZE / 2:
                        dev_class.append(1)
                        linenum.add(ln)
                else:
                    if zeros < TRAIN_SIZE / 2:
                        dev_class.append(0)   
                        linenum.add(ln)
                        zeros += 1    
                ln += 1
            else:
                break
    print len(dev_class)
    ln = 0
    
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        if len(dev_train) < TRAIN_SIZE:
            if ln in linenum:
                l1, p1 = feature_evalutaion(h1, ref) 
                l2, p2 = feature_evalutaion(h2, ref)
                sm1 = simple_meteor(h1, ref)
                sm2 = simple_meteor(h2, ref)
                fts = [l1, l2, p1, p2, sm1, sm2]
                dev_train.append(fts)  
        else:
            break
        ln += 1
    
    print len(dev_train)
    print "Finish preparation"
    print "Start to train..."
    clf = svm.SVC()
    clf.fit(dev_train, dev_class)
    print "Training is finished"
    
    # predict
    f = open("train.out", 'w')
    print "Predicting on..."
    
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        l1, p1 = feature_evalutaion(h1, ref) 
        l2, p2 = feature_evalutaion(h2, ref)
        sm1 = simple_meteor(h1, ref)
        sm2 = simple_meteor(h2, ref)
        
        fts = [l1, l2, p1, p2, sm1, sm2]
        ans = clf.predict([fts])
        f.write(str(ans[0]) + "\n")
        # if ans[0] == 0:
        #     f.write("0\n")
        # else:
        #     sore1 = sm1 + 0.3 * (0.5 * p1 + l1)
        #     sore2 = sm2 + 0.3 * (0.5 * p2 + l2)
        #     if sore1 == sore2:
        #         f.write("0\n")
        #     elif sore1 < sore2:
        #         f.write("-1\n")
        #     else:
        #         f.write("1\n")
        
    f.close()
    
   
    # print right
if __name__ == '__main__':
    main()