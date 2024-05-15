from typing import List, Set, Tuple

from gensim.models import KeyedVectors
import numpy as np
from embedding_task.embedding_task import EmbeddingTask


class GensimEmbeddingTask(EmbeddingTask):
    def __init__(self, **config_kwargs):
        """
        Params:
          config_kwargs: Config variable may contain the following information:
            emb_file_source: filepath of the embedding file.
            binary: bool to check whether the embedding file is binary or not.
            vocabulary_file_source: filepath that contains the vocabulary.
        Returns:
        """
        emb_file_source = config_kwargs['data_source']
        binary = config_kwargs['binary']
        vocabulary_file_source = config_kwargs['vocabulary'] if 'vocabulary' in config_kwargs else None
        
        super().__init__(vocabulary_file_source=vocabulary_file_source)
        self.model = self.read_embedding(emb_file_source, binary)
    
    @staticmethod
    def read_embedding(emb_file_source: str, binary: bool) -> KeyedVectors:
        model = KeyedVectors.load_word2vec_format(emb_file_source, binary=binary)
        return model
    
    def get_word_vectors(self):
        return self.word_vectors
    
    def execute(self) -> Tuple[np.ndarray, List[str]]:
        self.word_vectors = []
        new_vocabulary = []
        for word in self.vocabulary:
            if word in self.model:
                self.word_vectors.append(np.asarray(self.model[word], dtype=np.float32))
                new_vocabulary.append(word)

        self.vocabulary = new_vocabulary.copy()
        self.word_vectors = np.array(self.word_vectors)
        
        return self.word_vectors, self.vocabulary

    
    