# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import re, os
import time
import numpy as np 
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

WnL = WordNetLemmatizer() # Lemmatizer instance
LS = LancasterStemmer() # Stemmer instance

def load_stop_words(path):
    """ return stop words in a python list """
    with open(path, 'rb') as f:
        stopwords = f.read()
    return stopwords.split('\r\n')

def load_docs_ap(path):
    """ read ap.txt file"""
    begin = False
    docs = []
    num_docs = 0
    with open(path, 'r') as f:
        for line in f:
            if line == "<TEXT>\n":
                begin = True
                docs.append("")
                continue
            if line == " </TEXT>\n":
                begin = False
                num_docs += 1
                continue
            if begin:
                docs[num_docs] += line
    
    return docs

def load_20newsgroups(path, num_docs=100):
    """ 
    arguments
    ----------
    path is the root directory which contains all newsgroup, 
    num_docs is number of documents in each label we want to select

    return
    -------
    docs:list - a list contains all documents
    labels:list - labels of all documents ordered as docs
    """
    labels_folder = os.listdir(path)
    docs = []
    labels = []
    docs_id = []
    i = 0
    for label in labels_folder:
        path2files = path + '/' + label
        files = os.listdir(path2files)
        numdoc = 0
        for fn in files:
            if numdoc == num_docs: break
            numdoc += 1
            with open(path2files + '/' + fn, 'rb') as f:
                docs.append(f.read())
                labels.append(i)
                docs_id.append(fn)
        i += 1
    
    return docs, labels, docs_id

def normalize(doc, stopwords = None, lemma = False, stem= False):
    """
    Parameters
    -----------
    - doc:str - is a document that needs to be normalized
    - stopwords:list - if it is supplied, the document will be eliminated all the stopwords
    - lemmatize:bool - if it is True, all the word will be converted into lemma form

    Returns
    -------
    - doc:str - normalized document
    """
    # change currency sign followed by number to ' currency '
    # doc = re.compile(r'(\€|\¥|\£|\$)\d+([\.\,]\d+)*').sub(' currency ', doc )

    # change hh:mm:ss to " timestr "
    # doc = re.compile(r'(\d{2}):(\d{2}):(\d{2})').sub(' timestr ', doc)

    # # change email to ' emailaddr '
    # doc = re.compile(r'[^\s]+@[^\s]+').sub(' emailaddr ', doc)

    # # change link to ' urllink '
    # doc = re.compile(r'(((http|https):*\/\/[^\s]*)|((www)\.[^\s]*)|([^\s]*(\.com|\.co\.uk|\.net)[^\s]*))').sub(' urllink ', doc)

    # change phone number into ' phone_numb '
    # doc = re.compile(r'\(?([0-9]{3})\)?([ .-]?)([0-9]{3})\2([0-9]{4})').sub(' phonenumb ', doc)

    # change sequence of number to ' numb_seq '
    # doc = re.compile(r'\d+[\.\,]*\d*').sub(' numbseq ', doc)

    # lowercase and split doc by characters are not in  0-9A-Za-z
    docArr = re.compile(r'[^a-zA-Z0-9]').split(doc.lower())
    
    # docArr = []
    # for sent in sent_tokenize(doc.lower()):
    #     for word in word_tokenize(sent):
    #         docArr.append(word)

    if lemma:
        docArr = [lemmatize(word) for word in docArr]

    if stem:
        docArr =[LS.stem(word) for word in docArr]
        
    # remove stopwords
    if stopwords:
        stopwords = set(stopwords)
        docArr = [word for word in docArr if ((word not in stopwords) and (word != ''))]

    # return
    doc = ' '.join(docArr)
    return doc

def lemmatize(word):
    """
    transform word into lemma form
    ```
        { Part-of-speech constants
        ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
        }
    ```
    """
    rootWord = WnL.lemmatize(word, pos='n')
    if rootWord == word:
        rootWord = WnL.lemmatize(word, pos='a')
        if rootWord == word:
            rootWord = WnL.lemmatize(word, pos='v')
            if rootWord == word:
                rootWord = WnL.lemmatize(word, pos='r')
                if rootWord == word:
                    rootWord = WnL.lemmatize(word, pos='s')
    return rootWord

def stats_and_make_BoW(normalized_docs, max_df = 1.0, min_df = 0.0):
    """
    Parameters:
    ----------
    - `normalized_docs:list` - a list that contains all documents which are normalized
    - `max_df:float` - prune words that appear too common over all documents (not help us to distinguish docs)
    - `min_df:float` - prune words that appear too seldom over all documents (seem not appear in the future)
    
    Returns :
    -------
    - `bag:list` - bag of words
      * a list with all words are pairwise different (Bag Of Word),
    all the words in stats_all that has DF(word) value (the number of documents 
    has that word) not too big (not too popular in dataset since it is less 
    important for differentiate documents)

    - `stats_in_docs:list`
      * `stats_in_doc[i][word] = freq`, is the frequency of the word in document i'th
      * *NOTE*: `i` depends on the order of the document in list all documents

    - `dfs:dict`
      * `dfs[word] = number`, (document frequency) is the number of documents has that word
    """
    
    stats_in_doc = []
    for i in range(len(normalized_docs)):
        doc = normalized_docs[i]
        stats_in_doc.append({})
        for word in doc.split(' '):
            # Increase the frequency of the word in a documents
            if stats_in_doc[i].has_key(word):
                stats_in_doc[i][word] += 1
            else:
                stats_in_doc[i][word] = 1
    
    # Make stats_all dictionary
    dfs = {}
    for i in range(len(normalized_docs)):
        for (word, freq) in stats_in_doc[i].items():
            if dfs.has_key(word):
                dfs[word] += 1
            else:
                dfs[word] = 1 
    
    # create bag of words
    bag = []
    for word in dfs.keys():
        if dfs[word] <= max_df if (type(max_df) == int) else (len(normalized_docs) * max_df) \
        and dfs[word] >= min_df if (type(min_df) == int) else (len(normalized_docs) * min_df):
            bag.append(word)
    
    return bag, stats_in_doc

def standardize(pos, bag, stats_in_doc):
    """
    make sparse vector with the form:
    X[w] = freq - w is index of word in bag of word, freq is the frequency of that word in the document
    """
    X = {}
    for word in stats_in_doc[pos].keys():
        index = indexOf(bag, word)
        if index == -1:
            # print("word is not in the bag")
            continue
        if X.has_key(index):
            print("duplicated word! - " + word)
        else:
            X[index] = stats_in_doc[pos][word]
    
    return X

def indexOf(X, element):
    try:
        index = X.index(element)
        return index
    except:
        return -1

def run_gensim(normalized_docs):
    """
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=3, num_words=3))
    """
    normalized_docs = [doc.split(' ') for doc in normalized_docs]
    import gensim
    from gensim import corpora, models

    dictionary = corpora.Dictionary(normalized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in normalized_docs]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=10, num_words=5))

if __name__ == "__main__":
    print("START PROGRAM")

    time_stack = [time.time()]
    savedir = os.path.dirname(os.path.realpath(__file__)) + '/generated_files'
    # Preprocessing
    print("Loading data from file...")
    docs = load_docs_ap('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/topic model/LDA/lda-c/example/ap/ap.txt')
    stopwords = load_stop_words('/media/hieu/DATA/ShareOS/PythonVirtualEnv/WorkSpace/test/unsupervised_learning/k-mean/stopwords.txt')
    print("{} documents".format(len(docs)))
    print("{} words in stopwords".format(len(stopwords)))

    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))
    
    print("Normalizing all documents...")
    skip = False
    if skip:
        print("Loading from file...")
        docs = pickle.load(open(savedir + '/normalized_docs.data', 'rb'))
        print("{} documents".format(len(docs)))
    else:
        docs = [normalize(doc, stopwords, lemma=False, stem= False) for doc in docs]
        print("{} documents".format(len(docs)))
        print("writing to file...")
        pickle.dump(docs, open(savedir + '/normalized_docs.data', 'wb'))
    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))
    
    # print("press Enter to continue...")
    # a = raw_input()

    print("Making dictionary and statistic... ")
    skip = False
    if skip:
        print("Loading from file...")
        with open(savedir + '/BoW.data', 'r') as f:
            bag = f.readlines().split('\n')
        stats_in_doc = pickle.load(open(savedir + '/stats_in_doc.data'))
        print("{} words in bag of words".format(len(bag)))
    else:
        bag, stats_in_doc = stats_and_make_BoW(docs, max_df = 0.7, min_df = 5)
        print("{} words in bag of words".format(len(bag)))
        print("writing to file...")
        with open(os.path.join(savedir, 'BoW.data'), 'w') as f:
            for i in range(len(bag)):
                f.write(bag[i])
                if i != len(bag) - 1: f.write('\n')
        pickle.dump(stats_in_doc, open(os.path.join(savedir, 'stats_in_doc.data'), 'wb'))
    
    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))

    print("press Enter to continue...")
    a = raw_input()
    
    print("Making sparse vector for each document ...")
    skip = False
    if skip:
        print("Loading from file...")
        X = [{} for _ in range(len(docs))]
        i = 0
        with open(savedir + '/sparse_docs.data', 'r') as f:
            for line in f:
                line = line.split(' ')
                for term in line[1:]:
                    word_index, freq = term.split(':')
                    X[i][int(word_index)] = int(freq)
                i += 1
    else:
        print("standardizing and writing to file...")
        open(savedir + '/sparse_docs.data', 'w').close()
        X = [{} for _ in range(len(docs))]
        for i in range(len(docs)):
            X[i] = standardize(i, bag, stats_in_doc)
            num_terms = np.sum(X[i].values())
            with open(savedir + '/sparse_docs.data', 'a') as f:
                f.write(str(num_terms))
                for index in X[i].keys():
                    f.write(' ' + str(index) + ':' + str(X[i][index]))
                if i != len(docs) - 1: f.write('\n')
    
    time_stack.append(time.time())
    print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))

    print("Writing meta data...")
    with open(savedir + '/meta.data', 'w') as f:
        f.write("num_docs {}\n".format(len(docs)))
        f.write("num_terms {}\n".format(len(bag)))

    # print("Run gensim lda model...")
    # run_gensim(docs)

    # time_stack.append(time.time())
    # print("Done in {} seconds \n".format(time_stack[-1] - time_stack[-2]))
    