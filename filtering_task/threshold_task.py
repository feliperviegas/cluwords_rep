from typing import List
import numpy as np
from scipy.sparse import csr_matrix
from filtering_task.filtering_task import FilteringTask
from utils.incremental_coo_matrix import IncrementalCOOMatrix


class ThresholdTask(FilteringTask):
    def __init__(self, threshold_value: float):
        super().__init__()

        self.threshold_value = threshold_value

    def execute(self, knn_labels: np.array, knn_distances: np.array, vocabulary: List[str]) -> csr_matrix:
        n_words = len(vocabulary)
        self.similarity_matrix = IncrementalCOOMatrix(shape=(n_words, n_words), dtype=np.float32)
        
        for word_ref_index in range(0, n_words):
            words = []
            for index, k in enumerate(knn_labels[word_ref_index]):
                if 1 - knn_distances[word_ref_index][index] >= self.threshold_value:
                    words.append(vocabulary[k])
                    self.similarity_matrix.append(word_ref_index, k, round(1. - knn_distances[word_ref_index][index], 2))

        # Diagonal must be 1.0
        self.similarity_matrix = self.similarity_matrix.tocsr()
        for idx in range(self.similarity_matrix.shape[0]):
            self.similarity_matrix[idx, idx] = 1.0
            
        return self.similarity_matrix