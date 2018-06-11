#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from src.utils.newsgroup_dataprovider import TwentyNewsgroup
from src.utils.reuters_dataprovider import ReutersTfIdfVectors
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# https://arxiv.org/pdf/1607.01759.pdf
# https://statsbot.co/blog/text-classifier-algorithms-in-machine-learning/
from nltk.corpus import reuters

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
#     'comp.graphics'
# ])
dp = ReutersTfIdfVectors()

X_train = dp.fetch_dataset_train().data
X_test = dp.fetch_dataset_test().data

y_train = dp.fetch_dataset_train().target
y_train = np.argmax(y_train, axis=1)
# y_train = to_categorical(np.asarray(y_train))
y_test = dp.fetch_dataset_test().target
y_test = np.argmax(y_test, axis=1)
# y_test = to_categorical(np.asarray(y_test))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
print(X_train.shape)

#lsvc = LinearSVC(C=0.05, penalty="l2", dual=False).fit(X_train, y_train)
#model = SelectFromModel(lsvc, prefit=True)
#X_train = model.transform(X_train)

model = SelectKBest(chi2, k=10).fit(X_train, y_train)
X_train = model.transform(X_train)

print(X_train.shape)

X_test = model.transform(X_test)

clf = MultinomialNB(alpha=.01)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
# accuracy = np.mean(predicted == y_test)
# print('Accuracy: %f' % (accuracy * 100))
print('MNB Accuracy: %f' % (100 * accuracy_score(predicted, y_test)))

# Logistic Regression
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
print('LogReg Accuracy: %f' % (100 * accuracy_score(y_test, pred)))
