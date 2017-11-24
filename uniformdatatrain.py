#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from sklearn import svm
# from nltk.corpus import wordnet as wn
from nltk import pos_tag

def word_matches(h, ref):
    return sum(1 for w in h if w in ref)

def string_match(h1, h2, ref, n):
    h1_ngram = []
    h2_ngram = []
    ref_ngram = []
    for i in range(len(h1) - n + 1):
        w = ""
        for j in range(n):
             w += h1[i + j]
        h1_ngram.append(w)
    for i in range(len(h2) - n + 1):
        w = ""
        for j in range(n):
             w += h2[i + j]
        h2_ngram.append(w)
    
    for i in range(len(ref) - n + 1):
        w = ""
        for j in range(n):
             w += ref[i + j]
        ref_ngram.append(w)
    
    rset = set(ref_ngram)
    h1_match = word_matches(h1_ngram, rset)
    h2_match = word_matches(h2_ngram, rset)
    
    p1 = h1_match * 1.0 / len(rset)
    r1 = h1_match * 1.0 / len(h1_ngram)
    f1 = 0.0
    if p1 + r1 != 0:
        f1 = 2 * p1 * r1 / (p1 + r1)

    p2 = h2_match * 1.0 / len(rset)
    r2 = h2_match * 1.0 / len(h2_ngram)
    f2 = 0.0
    if p2 + r2 != 0:
        f2 = 2 * p2 * r2 / (p2 + r2)

    metrics = [p1 - p2, r1 - r2, f1 - f2]
    return metrics

def pos_match(l, h1_pos, h2_pos, ref_pos):
    h1 = []
    h2 = []
    ref = []
    for pos in h1_pos:
        h1.append(pos[1])
    for pos in h2_pos:
        h2.append(pos[1])
    for pos in ref_pos:
        ref.append(pos[1])
    for i in range(5):
        if i > len(h1) or i > len(h2) or i > len(ref):
            l.append(0)
            l.append(0)
            l.append(0)
            continue
        metrics = string_match(h1, h2, ref, i)
        l.append(metrics[0])
        l.append(metrics[1])
        l.append(metrics[2])


def feature_extraction(h1, h2, ref):
    l = []
    # word count
    h1_match = word_matches(h1, ref)
    h2_match = word_matches(h2, ref)
   
    l.append((h1_match - h2_match) * 1.0 / len(ref))
   
    # String match
    ave_p = 0.0
    for i in range(2, 5):
        if i > len(h1) or i > len(h2) or i > len(ref):
            l.append(0)
            l.append(0)
            l.append(0)
            continue
        metrics = string_match(h1, h2, ref, i)
        l.append(metrics[0])
        l.append(metrics[1])
        l.append(metrics[2])
        ave_p += metrics[0]
    l.append(ave_p / 3)
    # POS
    h1_pos = pos_tag(h1)
    h2_pos = pos_tag(h2)
    ref_pos = pos_tag(ref)
    pos_match(l, h1_pos, h2_pos, ref_pos)
    # similarity

    return l


def main():

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
    TRAIN_SIZE = 3333
    
    dev_train = []
    dev_class = []
    
   
    # Prefind
    line_num0 = set([])
    line_num1 = set([])
    line_numN1 = set([])
    
    ln = 1
    with open(opts.golden) as f:  
        for line in f:
            cl = int(line)
            if cl == 0 and len(line_num0) < TRAIN_SIZE:
                line_num0.add(ln)
                dev_class.append(cl)
            elif cl == 1 and len(line_num1) < TRAIN_SIZE:
                line_num1.add(ln)
                dev_class.append(cl)
            elif cl == -1 and len(line_numN1) < TRAIN_SIZE:
                line_numN1.add(ln)
                dev_class.append(cl)
            
            if len(line_num0) == TRAIN_SIZE and len(line_num1) == TRAIN_SIZE and len(line_numN1) == TRAIN_SIZE:
                break
            ln += 1

   
    ln = 1
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        if ln in line_num0 or ln in line_num1 or ln in line_numN1:
            fts = feature_extraction(h1, h2, ref)
            dev_train.append(fts)
        elif len(dev_train) > TRAIN_SIZE * 3:
            break
        ln += 1
    
            

    print "Finish preparation"
    print len(dev_class)
    print len(dev_train)
    
    print "Start to train..."
    clf = svm.SVC()
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
        f.write(str(ans[0]) + "\n")
   
    print right

# def tmp():
#     f = open("eval.out", 'w')
#     f.write(str(1) + "\n")
#     f.close()

if __name__ == '__main__':
    main()