import numpy as np
import pandas as pd
from task.task import Task


# WeightingTask inherits from Task
class WeightingTask(Task):
    def __init__(self):
        super().__init__()
        
    def execute(self, data: pd.DataFrame, semantic_matrix: np.array) -> np.array:
        # Implement weighting logic here
        print("Performing weighting")

        if semantic_matrix:
            raise AttributeError
        
        if data:
            raise AttributeError
        
        self.data = data
        self.semantic_matrix = semantic_matrix

        # TODO implement TFIDF dot matrix.

        return 
