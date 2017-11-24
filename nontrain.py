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
    for i in range(1, 5):
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
    for i in range(1, 5):
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

   
    ln = 0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):       
        fts = feature_extraction(h1, h2, ref)
        ans = sum(fts)
        if ans > 0.5:
            print 1
        elif ans < -0.5:
            print -1
        else:
            print 0
  

# def tmp():
#     f = open("eval.out", 'w')
#     f.write(str(1) + "\n")
#     f.close()

if __name__ == '__main__':
    main()