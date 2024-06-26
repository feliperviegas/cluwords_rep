from typing import List
import numpy as np
import hnswlib

from clustering_task.clustering_task import ClusteringTask
# import pandas as pd

# from utils.incremental_coo_matrix import IncrementalCOOMatrix


class HnswTask(ClusteringTask):
    def __init__(self, **config_kwargs):
        """
        Params:
          config_kwargs: Config variable may contain the following information:
            n_threads: Number of threads to use in the HNSW method (deafult 1).
            num_neighbors: Number of neighbors (default 10).
        Returns:
        """
        n_threads = config_kwargs['n_threads'] if 'n_threads' in config_kwargs else 1
        num_neighbors = config_kwargs['num_neighbors'] if 'num_neighbors' in config_kwargs else 10
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

