import numpy as np
from keras.layers import Embedding, Dense, Flatten, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer


class KerasConv1D:
    def __init__(self, train_dataset, embedding_matrix, vec_dim, max_seq_len, max_num_words, num_cat):
        self.vec_dim = vec_dim
        self.max_seq_len = max_seq_len
        self.max_num_words = max_num_words
        self.num_cat = num_cat

        tokenizer = Tokenizer(num_words=self.max_num_words)
        tokenizer.fit_on_texts(train_dataset)
        self.word_index = tokenizer.word_index

        self.embedding_matrix = embedding_matrix
        self.keras_model = self.__build_model()

    # def __build_embedding_matrix(self):
    #     embedding_matrix = np.zeros((len(self.word_index) + 1, self.vec_dim))
    #     for word, i in self.word_index.items():
    #         embedding_vector = self.vector_model[word]
    #         if embedding_vector is not None:
    #             embedding_matrix[i] = embedding_vector
    #     return embedding_matrix

    def __build_model(self):
        embedding_layer = Embedding(input_dim=len(self.word_index) + 1, output_dim=self.vec_dim,
                                    weights=[self.embedding_matrix],
                                    trainable=False, input_length=self.max_seq_len)

        keras_model = Sequential()
        keras_model.add(embedding_layer)
        keras_model.add(Conv1D(64, 5, activation='relu'))
        keras_model.add(MaxPooling1D(5))
        keras_model.add(Conv1D(64, 5, activation='relu'))
        keras_model.add(MaxPooling1D(5))
        keras_model.add(Conv1D(64, 5, activation='relu'))
        keras_model.add(MaxPooling1D(35))
        keras_model.add(Flatten())
        keras_model.add(Dense(32, activation='relu'))
        keras_model.add(Dense(self.num_cat, activation='softmax'))
        keras_model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
        return keras_model
