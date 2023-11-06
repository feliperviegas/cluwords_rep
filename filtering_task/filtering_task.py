from typing import List
from scipy.sparse import csr_matrix
import numpy as np
from task.task import Task


class FilteringTask(Task):
    def __init__(self):
        super().__init__()

    def execute(self, knn_labels: np.array, knn_distances: np.array, vocabulary: List[str]) -> csr_matrix:
        pass