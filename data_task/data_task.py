import pandas as pd
import spacy
from task.task import Task


# DataTask inherits from Task
class DataTask(Task):
    def __init__(self):
        super().__init__()
        self._nlp = spacy.load("en_core_web_sm")
        self._nlp.add_pipe("emoji", first=True)
    
    def execute(self, data_source: str) -> pd.DataFrame:
        pass
    
    