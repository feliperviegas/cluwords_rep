from typing import List
from scipy.sparse import csr_matrix, save_npz
import pandas as pd
from spacy.tokens.token import Token

from task.task import Task


class WriterTask(Task):
    def __init__(self, cluwords_repr_path: str, data_path: str) -> None:
        super().__init__()
        self.cluwords_repr_path = cluwords_repr_path
        self.data_path = data_path
    
    @staticmethod
    def parse_text(doc: List[Token]) -> List[str]:
        return [word.text for word in doc]
        
    def execute(self, cluwords_repr: csr_matrix, data_df: pd.DataFrame):
        data_df["text"] = data_df["text"].apply(self.parse_text)
        
        data_df.to_parquet(self.data_path)
        save_npz(self.cluwords_repr_path, cluwords_repr)
