#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from src.utils.newsgroup_dataprovider import TwentyNewsgroup

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

MAX_NUM_WORDS = 2000
NUM_CATEGORIES = dp.categories_size()

# ----------------------
# prepare dataset:
# ----------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(dp.fetch_dataset_train().data)
word_index = tokenizer.word_index

X_train = tokenizer.texts_to_matrix(dp.fetch_dataset_train().data, mode='tfidf')
X_test = tokenizer.texts_to_matrix(dp.fetch_dataset_test().data, mode='tfidf')

y_train = dp.fetch_dataset_train().target
y_train = to_categorical(np.asarray(y_train))
y_test = dp.fetch_dataset_test().target
y_test = to_categorical(np.asarray(y_test))

model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, 50, input_length=MAX_NUM_WORDS))
model.add(Dropout(0.25))
model.add(SimpleRNN(50, return_sequences=True))
model.add(Flatten())
model.add((Dense(NUM_CATEGORIES, activation='softmax')))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, epochs=10, batch_size=128)
loss, accuracy = model.evaluate(X_test, y_test, batch_size=128)
print('Accuracy 3: %f' % (accuracy * 100))
