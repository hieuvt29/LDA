from __future__ import division, print_function, unicode_literals
import numpy as np 
from scipy.special import gamma, digamma, gammaln, psi
from scipy.sparse import coo_matrix
from utils import *
import os
import pickle
import time

class LDA(object):
    def __init__(self, K = 1, V = 1, alpha = 0.1, beta = None, docs = None, method=None, loops=None, run_em=False, em_iters = 0, doc_iters = 0, em_converge = 1e-4, doc_converge = 1e-6):
        """
        K - number of topics
        V - number of distint terms in corpus
        method  
            random: random will initialize beta randomly based on data , data is required
            loaded: load from file, loops is required to specify which version we want to load
        """
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.K = K
        self.V = V
        self.em_converge = em_converge
        self.doc_converge = doc_converge

        if method == "random":
            self.initBeta()
            
        if method == "loaded":
            self.load(loops)

        if run_em:
            self.em_alg(em_iters = em_iters, doc_iters = doc_iters)

    def initBeta(self):
        self.beta = np.zeros((self.K, self.V))
        num_doc_per_topic = 2

        for i in range(num_doc_per_topic):
            rand_index = np.random.permutation(len(self.docs)).tolist()
            for k in range(self.K):
                d = rand_index[k]
                doc = self.docs[d]
                for n in doc.keys():
                    self.beta[k][n] += doc[n]
            
        self.beta += 1

        self.beta = self.beta / np.sum(self.beta, axis=1).reshape(self.K, 1)


    def em_alg(self, em_iters = 10, doc_iters = 10):
        self.gamma = np.zeros((len(self.docs), self.K))
        lb_old = 0
        isConverge = False
        it = 0
        while it < em_iters:
            lb = 0
            print("EM {} ...".format(i))
            st = time.time()
            phi_sum = np.zeros((self.K, self.V))
            for (d, doc) in enumerate(self.docs):
                doc_lb, var_phi, var_gamma = self.inference(doc, iters = doc_iters)
                self.gamma[d] = var_gamma

                Nd = len(doc.keys())
                row = range(Nd)
                col = doc.keys()
                data = doc.values()
                ind = coo_matrix((data, (row, col)), shape=(Nd, self.V)) #counter_indicator matrix (Nd, V)

                phi_sum += var_phi.T.dot(ind.toarray()) # (Nd, K).T x (Nd, V) = (K, V)
                
                lb += doc_lb
                if (d % 500 == 0 and d != 0):
                    print("doc : ", d)
    
            self.beta = phi_sum / np.sum(phi_sum, axis=1).reshape(self.K, 1)
            
            changed = lb_old - lb
            print("Lowerbound: {}, changed: {}".format(lb, changed))
            if ( np.fabs(changed / lb_old) <= self.em_converge):
                isConverge = True
            else:
                lb_old = lb
            print("------ Run in: {} s ------".format(time.time() - st))

            if (i == em_iters - 1) or isConverge:
                self.save(state = "final")
                return
            elif (i + 1) % 5 == 0:
                self.save(state= str(i + 1))
            it += 1
        
        self.em_iters_ran =  it + 1

    def inference(self, doc, iters = 10):
        Nd = len(doc.keys())
        K = self.K
        # init phi and gamma
        phi = np.ones((Nd, K)) * 1/K
        gamma = np.ones(K) * (np.sum(doc.values())/K + self.alpha)
        old_doc_lb = 0
        i = 0
        while i < iters:
            phi = self.beta[:, doc.keys()].T * np.exp(digamma(gamma)) # (K, Nd).T * (K, ) = (Nd, K)
            phi = phi / np.sum(phi, axis=1).reshape(Nd, 1)

            gamma = self.alpha + phi.T.dot(np.array(doc.values())) # (Nd, K).T * (Nd, ) = (K, )
            
            doc_lb = self.doc_lowerbound(phi, gamma, doc)
            changed = old_doc_lb - doc_lb
            if (np.fabs(changed / old_doc_lb) <= self.doc_converge):
                break
            else:
                old_doc_lb = doc_lb
                i += 1
        # print("{} iterations, lowerbound: {}".format(i, doc_lb))

        return doc_lb, phi, gamma

    def doc_lowerbound(self, phi, gamma, doc):
        gamma_sum = np.sum(gamma)
        sub_digamma = digamma(gamma) - digamma(gamma_sum) # (K, ) - (,) = (K, )
        counts = np.array(doc.values()).reshape(len(doc.keys()), 1) # (Nd, 1)

        l1 = (self.alpha - 1) * np.sum(sub_digamma) # float 
        # l2 = np.sum(phi * sub_digamma * counts) # (Nd, K) * (K, ) * (Nd, 1)
        # l3 = np.sum(phi * np.log(self.beta[:, doc.keys()].T) * counts) # (Nd, K) * (K, Nd).T = (Nd, K) * (Nd, 1)
        l4 = - gammaln(gamma_sum) + np.sum(gammaln(gamma)) - (gamma - 1).T.dot(sub_digamma)
        #float + sum((K, )) - (K, ).Tx(K, ) = float
        # l5 = - np.sum(phi * np.log(phi) * counts) # (Nd, K) * (Nd, K) * (Nd, 1)
        l6 = np.sum(counts * np.nan_to_num(phi * (sub_digamma + np.log(self.beta[:, doc.keys()].T) - np.log(phi))))
        # (Nd, K) * ( (K, ) + (K, Nd).T - (Nd, K) = (Nd, K) ) * (Nd, 1)
        l = l1 + l4 + l6
        # print("l1: {}, l4: {}, l6: {}".format(l1,l4,l6))
        # print("lb: ", l)
        # a = raw_input()
        return l

    def save(self, state):
        savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'
        pickle.dump(self.gamma, open(savedir + '/gamma-' + state + '.K_' + str(self.K) + '.V_' + str(self.V), 'wb'))
        pickle.dump(self.beta, open(savedir + '/beta-' + state + '.K_' + str(self.K) + '.V_' + str(self.V), 'wb'))
        pass

    def load(self, loops):
        savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'        
        self.gamma = pickle.load(open(savedir + '/gamma-'+ str(loops) +'.K_' + str(self.K) + '.V_' + str(self.V), 'rb'))
        self.beta = pickle.load(open(savedir + '/beta-'+ str(loops) +'.K_' + str(self.K) + '.V_' + str(self.V), 'rb'))
        pass

    def print_topics(self, vocab, nwords = 10, verbose=False):
        indices = range(len(vocab))
        topic_no = 0
        savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'  
        f = open(savedir + '/topic_words.txt', 'w')
        for topic in self.beta:
            # print('topic {}'.format(topic_no))
            f.write('topic {}  '.format(topic_no))
            indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
            if verbose:
                topic = -np.sort(-topic)
                for i in range(nwords):
                    # print('  {} * {}'.format(vocab[indices[i]],topic[i]), end=",")
                    f.write('  {} * {},'.format(vocab[indices[i]],topic[i]))
            else:
                for i in range(nwords):
                    # print('  {}'.format(vocab[indices[i]]), end=",")
                    f.write('  {},'.format(vocab[indices[i]]))
            topic_no = topic_no + 1
            # print("\n")
            f.write("\n")
        f.close()

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
    ldaModel = LDA(K = K, V = V, alpha = alpha, docs = X, method="random", run_em=True, em_iters = em_iters, doc_iters= doc_iters)

    vocab = []
    # with open(savedir + '/BoW.data', 'r') as f:
    #     for line in f: vocab.append(line[:-1])
    with open('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-c/example/ap/vocab.txt', 'r') as f:
        for line in f: vocab.append(line[:-1])

    ldaModel.print_topics(vocab = vocab, nwords = 10, verbose = False)

    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[0]))
    with open(savedir + '/logs', 'a') as f:
        f.write("{} terms, {} topics, {} EM iterations, {} doc iterations, {} alpha: {} seconds\n".format(V, K, ldaModel.em_iters_ran, doc_iters, alpha, time_stack[-1] - time_stack[0]))