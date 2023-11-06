
import pandas as pd

from task.task import Task


# ClusteringTask inherits from Task
class ClusteringTask(Task):
    def __init__(self, num_neighbors):
        super().__init__()

        self.num_neighbors = num_neighbors

    def execute(self, data: pd.DataFrame):
        # Implement clustering logic here
        print(f"Performing clustering with {self.num_neighbors} neighbors")

        if data.isinstance(pd.DataFrame):
            raise AttributeError
        
        self.data = data

        # TODO implment clustering step