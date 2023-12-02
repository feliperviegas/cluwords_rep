from typing import List
import pandas as pd
import spacy
from task.task import Task


class EmbeddingTask(Task):
    def __init__(self, vocabulary_file_source: str = None):
        super().__init__()
        self._nlp = spacy.load("en_core_web_sm")
        self._nlp.add_pipe("emoji", first=True)
        
        if vocabulary_file_source:
            self.vocabulary = self.read_vocabulary(vocabulary_file_source)
        else:
            self.vocabulary = list(set([word.lower() for word in list(self._nlp.vocab.strings)]))
            
    
    @staticmethod
    def read_vocabulary(vocabulary_file_source) -> List[str]:
        vocabulary_pd = pd.read_csv(vocabulary_file_source, names=["word"])
        vocabulary = list(set([str(word).lower() for word in vocabulary_pd["word"].to_list()]))
        return vocabulary
    
    def get_vocabulary(self):
        return self.vocabulary

    def execute(self):
        pass