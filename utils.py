import numpy as np

def max_term(docs):
    res = 0
    for doc in docs:
        num_words = np.sum(doc.values())
        if res < num_words: res = num_words
    
    return res