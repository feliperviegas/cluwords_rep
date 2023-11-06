from typing import List
import numpy as np
import hnswlib

from clustering_task.clustering_task import ClusteringTask
# import pandas as pd

# from utils.incremental_coo_matrix import IncrementalCOOMatrix


class HnswTask(ClusteringTask):
    def __init__(self, n_threads: int, num_neighbors: int):
        super().__init__(num_neighbors=num_neighbors)  
        self.n_threads = n_threads
        
    def execute(self, word_vectors: np.ndarray, vocabulary: List[str]):
        n_words = len(vocabulary)
        
        # Declaring index
        p = hnswlib.Index(space='cosine', dim=word_vectors.shape[1])  # possible options are l2, cosine or ip

        # Initializing index - the maximum number of elements should be known beforehand
        p.init_index(max_elements=n_words, ef_construction=200, M=64)
        
        word_vectors_labels = np.arange(n_words)
        # Element insertion (can be called several times):
        p.add_items(word_vectors, word_vectors_labels)

        # Controlling the recall by setting ef:
        p.set_ef(600)  # ef should always be > k

        p.set_num_threads(self.n_threads)

        # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
        self.labels, self.distances = p.knn_query(word_vectors, k=self.num_neighbors)
        
        return self.labels, self.distances
    
        # FILTERING STUFF
        
         # if self.threshold is None:
        #     cosine_vec = (1. - self.distances).flatten()
        #     self.threshold = np.round(np.percentile(cosine_vec, 95), 2)

        # self.similarity_matrix = IncrementalCOOMatrix(shape=(self.n_words, self.n_words), dtype=np.float32)
        # knn_list_of_dicts = []
        # for word_ref_index in range(0, self.n_words):
        #     words = []
        #     for index, k in enumerate(self.labels[word_ref_index]):
        #         if 1 - self.distances[word_ref_index][index] >= self.threshold:
        #             words.append(self.vocabulary[k])
        #             similarity_matrix.append(word_ref_index, k, round(1. - self.distances[word_ref_index][index], 2))

        #     knn_list_of_dicts.append(
        #         {
        #             "word_ref": self.vocabulary[word_ref_index],
        #             "word_neigh": words,
        #         }
        #     )

        # self.knn_words_df = pd.DataFrame(knn_list_of_dicts)

        # # Diagonal must be 1.0
        # similarity_matrix = similarity_matrix.tocsr()
        # for idx in range(similarity_matrix.shape[0]):
        #     similarity_matrix[idx, idx] = 1.0
            
        # return similarity_matrix
