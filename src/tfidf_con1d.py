#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from gensim.models import Word2Vec
from keras import activations
from keras.layers import Embedding, Dense, Flatten, Dropout, Activation, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from src.utils.newsgroup_dataprovider import TwentyNewsgroup
from keras.utils.np_utils import to_categorical

from src.keras_models.Keras_Conv1D import KerasConv1D
from src.embedding_matrix.gensim_w2v import W2V_Embedding
from src.embedding_matrix.sklearn_tfidf import TfIdf_Embedding

# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
# https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# https://arxiv.org/pdf/1607.01759.pdf
# https://statsbot.co/blog/text-classifier-algorithms-in-machine-learning/


# ----------------------
# Fetch train dataset:
# ----------------------
# dp = TwentyNewsgroup(categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med'])

dp = TwentyNewsgroup(categories=['alt.atheism',
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


VEC_DIM = 128
# MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 2000
NUM_CATEGORIES = dp.categories_size()

# ----------------------
# prepare dataset:
# ----------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(dp.fetch_dataset_train().data)
word_index = tokenizer.word_index
# integer encode documents
X_train = tokenizer.texts_to_matrix(dp.fetch_dataset_train().data, mode='tfidf')
X_test = tokenizer.texts_to_matrix(dp.fetch_dataset_test().data,mode='tfidf')


y_train = dp.fetch_dataset_train().target
y_train = to_categorical(np.asarray(y_train))
y_test = dp.fetch_dataset_test().target
y_test = to_categorical(np.asarray(y_test))

keras_model = Sequential()
keras_model.add(Embedding(MAX_NUM_WORDS, 512, input_length=MAX_NUM_WORDS))
keras_model.add(Dense(64, activation='relu'))
keras_model.add(Flatten())
keras_model.add(Dense(NUM_CATEGORIES, activation='softmax'))
keras_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

print(keras_model.summary())
keras_model.fit(X_train, y_train, epochs=10, batch_size=128)
loss, accuracy = keras_model.evaluate(X_test, y_test, batch_size=128)
print('Accuracy 3: %f' % (accuracy * 100))

