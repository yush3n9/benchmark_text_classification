#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

Feed w2v to mnb is not possible!
ValueError: Found array with dim 3. Estimator expected <= 2.

MNB requires input matrix with dim <=2

'''


import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.newsgroup_dataprovider import TwentyNewsgroup

# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# https://arxiv.org/pdf/1607.01759.pdf
# https://statsbot.co/blog/text-classifier-algorithms-in-machine-learning/


# ----------------------
# Fetch train dataset:
# ----------------------
# dp = TwentyNewsgroup(categories=[
#     'alt.atheism',
#     'comp.graphics',
#     'comp.os.ms-windows.misc',
#     'comp.sys.ibm.pc.hardware',
#     'comp.sys.mac.hardware',
#     'comp.windows.x',
#     'misc.forsale',
#     'rec.autos',
#     'rec.motorcycles',
#     'rec.sport.baseball',
#     'rec.sport.hockey',
#     'sci.crypt',
#     'sci.electronics',
#     'sci.med',
#     'sci.space',
#     'soc.religion.christian',
#     'talk.politics.guns',
#     'talk.politics.mideast',
#     'talk.politics.misc',
#     'talk.religion.misc'])
dp = TwentyNewsgroup(categories=[
    'alt.atheism',
    'comp.graphics'
    ])
VEC_DIM = 128
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
NUM_CATEGORIES = dp.categories_size()

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

# ----------------------
# build w2v model:
# ----------------------
w2v_model = Word2Vec(dp, iter=5, size=VEC_DIM, min_count=1)
vocab_size = len(w2v_model.wv.vocab)
weights = np.array(w2v_model.wv.syn0)
assert vocab_size == weights.shape[0], "vocab_size must be same as weights.shape[0]"
assert VEC_DIM == weights.shape[1], "vec_dim must be same as weights.shape[1]"

# ----------------------
# prepare dataset:
# ----------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(dp.fetch_dataset_train().data)
word_index = tokenizer.word_index


# ----------------------
# build embedding matrix:
# ----------------------
embedding_matrix = np.zeros((len(word_index) + 1, VEC_DIM))
for word, i in word_index.items():
    embedding_vector = w2v_model[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

X_train = pad_sequences(tokenizer.texts_to_sequences(dp.fetch_dataset_train().data), maxlen=MAX_SEQUENCE_LENGTH)
X_train = [embedding_matrix[i] for i in [l for l in X_train]]
print(X_train)

X_test = pad_sequences(tokenizer.texts_to_sequences(dp.fetch_dataset_test().data), maxlen=MAX_SEQUENCE_LENGTH)
X_test = [embedding_matrix[i] for i in [l for l in X_test]]
y_train = dp.fetch_dataset_train().target
# y_train = to_categorical(np.asarray(y_train))
y_test = dp.fetch_dataset_test().target
# y_test = to_categorical(np.asarray(y_test))


clf = MultinomialNB(alpha=.01)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
accuracy = np.mean(predicted == y_test)

print('Accuracy: %f' % (accuracy * 100))
