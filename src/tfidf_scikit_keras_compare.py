#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from src.utils.newsgroup_dataprovider import TwentyNewsgroup

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"


[word for word in word_list if word not in stopwords.words('english')]

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]


def print_top_features(tfidf_docs, features):
    for i, tfidf_doc in enumerate(tfidf_docs):
        print("Top words in document {}".format(i))
        #print(tfidf_doc)
        sorted_idx = np.argsort(tfidf_doc)[::-1]
        #print(sorted_idx)

        for idx in sorted_idx[:5]:
            print("\tWord: {}, \t\tTF-IDF: {}".format(features[idx], round(tfidf_doc[idx], 5)))


MAX_NUM_WORDS = 2000

# ----------------------
# TfIdf with keras
# ----------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_documents)
word_index = tokenizer.word_index
print(word_index)
idx_word = dict()
for k in word_index:
    idx_word[word_index[k]]=k
print(idx_word)

tfidf_keras = tokenizer.texts_to_matrix(all_documents, mode='tfidf')
print_top_features(tfidf_keras, idx_word)

# ----------------------
# TfIdf with scikit learn
# ----------------------
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
vectorizer = TfidfVectorizer()
tfidf_sklearn = vectorizer.fit_transform(all_documents)
#
print_top_features(tfidf_sklearn.toarray(), vectorizer.get_feature_names())


