import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf_Embedding:
    def __init__(self, data_provider, word_index, sublinear_tf, max_df, stop_words):
        self.dp = data_provider
        self.word_index = word_index
        self.sublinear_tf = sublinear_tf
        self.max_df = max_df
        self.stop_words = stop_words
        self.embedding = self.__build_embedding()

    def __build_embedding(self):
        vectorizer = TfidfVectorizer(sublinear_tf=self.sublinear_tf, max_df=self.max_df, stop_words=self.stop_words)
        tfidf = vectorizer.fit_transform(self.dp.fetch_dataset_train().data)

        idx_word = dict()
        for w in vectorizer.vocabulary_:
            idx_word[vectorizer.vocabulary_[w]] = w

        word_tfidf = dict()
        for idx, value in zip(tfidf.indices, tfidf.data):
            word_tfidf[idx_word[idx]] = value

        embedding_matrix = np.zeros((len(self.word_index) + 1, 1))
        for word, i in self.word_index.items():
            if word in word_tfidf:
                embedding_vector = word_tfidf[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        print(tfidf.shape)

        return embedding_matrix
