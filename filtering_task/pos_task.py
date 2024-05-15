import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import spacy
from filtering_task.filtering_task import FilteringTask


class PartOfSpeechTask(FilteringTask):
    def __init__(self, **config_kwargs):
        """
        Params:
          config_kwargs: Config variable may contain the following information:
            pos_filters: list of strings that contains the tags of spacy PoS.
            
        Returns:
        """
        pos_filters = config_kwargs['pos_filters'] if 'pos_filters' in config_kwargs else [] 
        
        super().__init__()
        self._nlp = spacy.load("en_core_web_sm")
        self._nlp.add_pipe("emoji", first=True)
        self.pos_filters = pos_filters
        
    def execute(self, semantic_matrix: csr_matrix, vocabulary: list[str]) -> csr_matrix:
        semantic_matrix = lil_matrix(semantic_matrix)
        for idx in range(semantic_matrix.shape[0]):
            word = vocabulary[idx]
            word_nlp = self._nlp(word)[0]
            if word_nlp.pos_ in self.pos_filters or word_nlp.is_stop:
                semantic_matrix[idx, :] = .0
                semantic_matrix[:, idx] = .0
                semantic_matrix[idx, idx] = 1.0
             
        return csr_matrix(semantic_matrix)
