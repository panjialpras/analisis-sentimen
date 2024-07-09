import pandas as pd
import numpy as np

def compute_tf(word_dict, bow):
    tf = {}
    bow_count = len(bow)
    for word, count in word_dict.items():
        tf[word] = count / float(bow_count)
    return tf

def compute_idf(doc_list):
    import math
    idf_dict = {}
    N = len(doc_list)

    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1
    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(N / float(val))
    return idf_dict

def compute_tfidf(tf_bow, idfs):
    tfidf = {}
    for word, val in tf_bow.items():
        tfidf[word] = val * idfs[word]
    return tfidf