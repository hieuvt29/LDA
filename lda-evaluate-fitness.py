from __future__ import division, print_function, unicode_literals
import numpy as np 
from scipy.special import gamma, digamma, gammaln, psi
from scipy.sparse import coo_matrix
from utils import *
from lda import LDA
import os
import pickle
import time

# def inference(ldaModel, doc, iters = 10):
#     Nd = len(doc.keys())
#     K = ldaModel.K
#     # init phi and gamma
#     phi = np.ones((Nd, K)) * 1/K
#     gamma = np.ones(K) * (np.sum(doc.values())/K + ldaModel.alpha)
#     old_doc_lb = 0
#     i = 0
#     while i < iters:
#         phi = ldaModel.beta[:, doc.keys()].T * np.exp(digamma(gamma)) # (K, Nd).T * (K, ) = (Nd, K)
#         phi = phi / np.sum(phi, axis=1).reshape(Nd, 1)

#         gamma = ldaModel.alpha + phi.T.dot(np.array(doc.values())) # (Nd, K).T * (Nd, ) = (K, )

#         doc_lb = ldaModel.doc_lowerbound(phi, gamma, doc)
#         changed = old_doc_lb - doc_lb
#         if (np.fabs(changed / old_doc_lb) <= ldaModel.doc_converge):
#             break
#         else:
#             old_doc_lb = doc_lb
#             i += 1
#     # print("{} iterations, lowerbound: {}".format(i, doc_lb))

#     return doc_lb, phi, gamma

# def doc_lowerbound(ldaModel, phi, gamma, doc):
#     gamma_sum = np.sum(gamma)
#     sub_digamma = digamma(gamma) - digamma(gamma_sum) # (K, ) - (,) = (K, )
#     counts = np.array(doc.values()).reshape(len(doc.keys()), 1) # (Nd, 1)

#     l1 = (ldaModel.alpha - 1) * np.sum(sub_digamma) # float 
#     # l2 = np.sum(phi * sub_digamma * counts) # (Nd, K) * (K, ) * (Nd, 1)
#     # l3 = np.sum(phi * np.log(ldaModel.beta[:, doc.keys()].T) * counts) # (Nd, K) * (K, Nd).T = (Nd, K) * (Nd, 1)
#     l4 = - gammaln(gamma_sum) + np.sum(gammaln(gamma)) - (gamma - 1).T.dot(sub_digamma)
#     #float + sum((K, )) - (K, ).Tx(K, ) = float
#     # l5 = - np.sum(phi * np.log(phi) * counts) # (Nd, K) * (Nd, K) * (Nd, 1)
#     l6 = np.sum(counts * np.nan_to_num(phi * (sub_digamma + np.log(ldaModel.beta[:, doc.keys()].T) - np.log(phi))))
#     # (Nd, K) * ( (K, ) + (K, Nd).T - (Nd, K) = (Nd, K) ) * (Nd, 1)
#     l = l1 + l4 + l6
#     # print("l1: {}, l4: {}, l6: {}".format(l1,l4,l6))
#     # print("lb: ", l)
#     # a = raw_input()
#     return l

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
    doc_iters = 40
    alpha = 0.1
    ldaModel = LDA(K = K, V = V, alpha = alpha, method="loaded", doc_iters= doc_iters)
    pre_prob = predictive_distribution(X[:100], 0.9, ldaModel)
    print(pre_prob)


        

    
