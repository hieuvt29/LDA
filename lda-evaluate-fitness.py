from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.special import gamma, digamma, gammaln, psi
from scipy.sparse import coo_matrix
from utils import *
from lda import LDA
import os, pickle, time
import matplotlib.pyplot as plt


def predictive_distribution(docs, ratio, ldaModel):
    predictive_prob = 0
    total_word_news = 0
    for (d, doc) in enumerate(docs):
        total_words = np.sum(doc.values())
        Nd = len(doc.keys())
        num_words = 0
        # split doc into observed + hold-out
        obs = {}
        ho = {}
        for w in doc.keys():
            if (np.sum(obs.values()) / total_words >= ratio):
                ho[w] = doc[w]
            else:
                obs[w] = doc[w]
        total_word_news += len(ho.keys())
        # inference
        lb, var_phi, var_gamma = ldaModel.inference(obs, iters = 40)
        mean_theta = var_gamma / var_gamma.sum()
        log_holdout_prob = 0
        for w_new in ho.keys():
            word_prob = 0
            for k in range(ldaModel.K):
                word_prob += mean_theta[k] * ldaModel.beta[k, w_new]
        
            log_holdout_prob += np.log(word_prob)

        predictive_prob += log_holdout_prob
    
    return predictive_prob / total_word_news

if __name__ == "__main__":
    time_stack = [time.time()]
    savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'
    print("Loading meta data...")
    # with open(savedir + '/meta.data', 'r') as f:
    #     for line in f:
    #         if line.split()[0] == "num_docs": 
    #             D = int(line.split()[1])
    #         if line.split()[0] == "num_terms": 
    #             V = int(line.split()[1])
    D = 2246
    V = 10473
    print("Loading sparse docs from file...")
    X = [{} for _ in range(D)]
    i = 0
    # with open(savedir + '/sparse_docs.data', 'r') as f:
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-c/example/ap/ap.dat', 'r') as f:
        for line in f:
            if line == '': continue
            line = line.split(' ')
            for term in line[1:]:
                word_index, freq = term.split(':')
                X[i][int(word_index)] = int(freq)
            i += 1
    
    print("Running lda algorithm...")
    K = 100
    em_iters = 100
    doc_iters = 20
    alpha = 0.1
    pre_prob_tests = []
    loopsArr = [5, 10, 15, 20, 25, 30, 35]
    for loops in loopsArr:
        print("Evaluate model with loops: ", loops)
        ldaModel = LDA(K = K, V = V, alpha = alpha, method="loaded", loops=loops, doc_iters= doc_iters)
        pre_prob = predictive_distribution(X[:100], 0.9, ldaModel)
        pre_prob_tests.append(pre_prob)
    print(pre_prob_tests)
    plt.figure()
    plt.plot(loopsArr, pre_prob_tests, '-b')
    plt.xlabel("Number of topics")
    plt.ylabel("Log Predictive Probability")
    plt.show()

    
