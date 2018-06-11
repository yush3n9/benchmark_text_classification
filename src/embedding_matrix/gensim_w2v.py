import numpy as np
from gensim.models import Word2Vec


class W2V_Embedding:
    def __init__(self, data_provider, word_index, iter, size, min_count):
        self.dp = data_provider
        self.word_index = word_index
        self.iter = iter
        self.size = size
        self.min_count = min_count
        self.embedding = self.__build_embedding()

    def __build_embedding(self):
        w2v_model = Word2Vec(self.dp, iter=self.iter, size=self.size, min_count=self.min_count)
        weights = np.array(w2v_model.wv.syn0)
        vocab_size = len(w2v_model.wv.vocab)
        assert vocab_size == weights.shape[0], "vocab_size must be same as weights.shape[0]"
        assert self.size == weights.shape[1], "vec_dim must be same as weights.shape[1]"

        embedding_matrix = np.zeros((vocab_size + 1, self.size))
        for word, i in self.word_index.items():
            embedding_vector = w2v_model[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
