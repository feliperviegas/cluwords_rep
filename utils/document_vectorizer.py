from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class DocumentVectorizer:
    def __init__(self):
        self.tf_vectorizer = None
        self.inverted_vocabulary = {}

    @staticmethod
    def read_raw_data(document_path):
        arq = open(document_path, 'r', encoding="utf-8")
        doc = arq.readlines()
        arq.close()
        documents = list(map(str.rstrip, doc))
        return documents

    def fit(self, documents, vocabulary):
        self.tf_vectorizer = CountVectorizer(max_features=len(vocabulary), vocabulary=vocabulary)
        self.tf_vectorizer.fit(documents)
        self.set_inverted_vocabulary()


    def transform(self, documents, dt=np.float32):
        if self.tf_vectorizer is not None:
            tf = self.tf_vectorizer.transform(documents)
            return np.asarray(tf.toarray(), dtype=dt)
        else:
            print("Not found Vectorizer object.")

    @staticmethod
    def get_binary(array):
        return (array > 0) * 1.

    def get_vocabulary(self):
        return self.tf_vectorizer.vocabulary_

    def set_inverted_vocabulary(self):
        for key, value in self.get_vocabulary().items():
            self.inverted_vocabulary[value] = key

    def get_inverted_vocabulary(self):
        return self.inverted_vocabulary

    def map_id_to_token(self, term_id: int):
        return self.inverted_vocabulary[term_id]

    @staticmethod
    def check_for_negative_values(array):
        sum_values = np.sum((array < 0) * 1.)
        if sum_values == 0:
            print("Documents have only positive values.")
        elif sum_values > 0:
            print("Documents have negative values.")
        else:
            print("Not sure what happened.")

    @staticmethod
    def as_numpy_array(list_data, dtype=np.int32):
        return np.asarray(list_data, dtype=dtype)

    def map_documents_to_tokens(self, documents) -> pd.Series:
        docs = []
        for document in documents:
            if isinstance(document, csr_matrix):
                _, cols = document.nonzero()
            else:
                _, cols = document.nonzero()
            tokens = []
            for term_id in cols:
                tokens.append(self.map_id_to_token(term_id))

            docs.append(tokens)

        docs = pd.Series(docs)
        return docs

