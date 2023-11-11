from typing import List, Set
import pandas as pd
import spacy
from task.task import Task


class EmbeddingTask(Task):
    def __init__(self, vocabulary_file_source: str = None):
        super().__init__()
        
        if vocabulary_file_source:
            self.vocabulary = self.read_vocabulary(vocabulary_file_source)
        else:
            nlp = spacy.load("en_core_web_sm")
            self.vocabulary = list(nlp.vocab.strings)
    
    @staticmethod
    def read_vocabulary(vocabulary_file_source) -> List[str]:
        pass
    
    def get_vocabulary(self):
        return self.vocabulary

    def execute(self):
        pass