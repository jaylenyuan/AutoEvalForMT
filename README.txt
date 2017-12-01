Run instruction:
python evaluate > eval.out

The feature extractions are implemented in nontrain.py

Feaatures or Metrics:
1. Simple METEOR with alpha = 0.2, with weight 1 to the total score
    Basic Metrics for evaluating machine translation.
2. 1-4 gram match F-score between hyp and reference, with weight 0.5 to the total
    This feature help measure the similarity between two sentence
3. 1-4 gram match F-score between pos-tag of hyps and that of reference, with weight 0.4 to the total
    According to the paper 'Regression and Ranking based Optimisation for Sentence Level Machine
Translation Evaluation', The string matching features and word count features only measure similarities on the lexical level,
but not over sentence structure or synonyms. So introduce the pos-tag matching.
    First tag two sentence with nltk.pos_tag, then do the match count.
    
