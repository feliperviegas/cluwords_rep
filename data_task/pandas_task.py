import pandas as pd

from data_task.data_task import DataTask


class PandasTask(DataTask):
    def __init__(self):
        super().__init__()
    
    def execute(self, data_source: str) -> pd.DataFrame:
        data = pd.read_csv(data_source)
        
        # text columns has the dtype "from spacy.tokens.token import Token"
        data['text'] = data['text'].apply(lambda text: [word for word in self._nlp(text)])
        
        return data