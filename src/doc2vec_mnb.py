#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

Feed w2v to mnb is not possible!
ValueError: Found array with dim 3. Estimator expected <= 2.

MNB requires input matrix with dim <=2

'''

# from gensim.models.doc2vec import LabeledSentence
from collections import namedtuple

import numpy as np
from gensim.models import doc2vec
from sklearn.naive_bayes import GaussianNB

from src.utils.newsgroup_dataprovider import TwentyNewsgroup

# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# https://arxiv.org/pdf/1607.01759.pdf
# https://statsbot.co/blog/text-classifier-algorithms-in-machine-learning/


# ----------------------
# Fetch train dataset:
# ----------------------
dp = TwentyNewsgroup(categories=[
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'])
# dp = TwentyNewsgroup(categories=[
#     'alt.atheism',
#     'comp.graphics',
#     'soc.religion.christian'
# ])
VEC_DIM = 128
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
NUM_CATEGORIES = dp.categories_size()
from sklearn import linear_model

# ----------------------
# build doc2vec:
# ----------------------
tagged_docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(dp.fetch_dataset_train().data):
    words = text.lower()
    tags = [i]
    tagged_docs.append(analyzedDocument(words, tags))

# doc2vec_model = doc2vec.Doc2Vec(tagged_docs, vector_size=160, window=10, min_count=7, workers=4)
doc2vec_model = doc2vec.Doc2Vec(tagged_docs, workers=4, dm=0, dbow_words=0, vector_size=200)

X_train = [doc2vec_model.infer_vector(d.lower()) for d in dp.fetch_dataset_train().data]
print(len(X_train))

X_test = [doc2vec_model.infer_vector(d.lower()) for d in dp.fetch_dataset_test().data]
print(len(X_test))
y_train = dp.fetch_dataset_train().target
# y_train = to_categorical(np.asarray(y_train))
y_test = dp.fetch_dataset_test().target
# y_test = to_categorical(np.asarray(y_test))


# clf = GaussianNB()
# clf.fit(X_train, y_train)

# g, p = [], []
# for i, v in enumerate(X_train):
#     predicted = doc2vec_model.docvecs.most_similar([v], topn=1)
#     p.append(y_train[predicted[0][0]])
#     g.append(y_train[i])
#
#     print(i, predicted[0])
# accuracy = np.mean(np.asarray(g) == np.asarray(p))
# print('Accuracy: %f' % (accuracy * 100))


# sm = [doc2vec_model.docvecs.most_similar([t], topn=1) for t in X_test]
# predicted = list()
# for s in sm:
#     predicted.append((y_train[s[0][0]]))
# print(predicted)
# accuracy = np.mean(np.asarray(predicted) == y_test)
# print('Accuracy: %f' % (accuracy * 100))

clf = GaussianNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
accuracy = np.mean(predicted == y_test)
print('GaussianNB Accuracy: %f' % (accuracy * 100))

clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
accuracy = np.mean(predicted == y_test)
print('LogisticRegression Accuracy: %f' % (accuracy * 100))
