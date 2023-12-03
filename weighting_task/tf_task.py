from typing import List
from spacy.tokens.token import Token
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from utils.document_vectorizer import DocumentVectorizer
from weighting_task.weighting_task import WeightingTask


class TfTask(WeightingTask):
    def __init__(self):
        super().__init__()
        self.document_vectorizer = DocumentVectorizer()
    
    @staticmethod
    def parse_str_list(texts: List[List[Token]]) -> List[List[str]]:
        texts_str = []
        for text in texts:
            text_str = text[0].text
            for idx in range(1, len(text)):
                text_str += f" {text[idx].text.lower()}"
            
            texts_str.append(text_str)
            
        return texts_str 
    
    @staticmethod
    def combine_bow_and_semantic_matrices(tf_repr: csr_matrix, semantic_matrix: csr_matrix) -> csr_matrix:
        return csr_matrix(tf_repr).dot(csr_matrix.transpose(semantic_matrix))
        
    def execute(self, data: pd.DataFrame, vocabulary: List[str], semantic_matrix: csr_matrix) -> np.array:
        texts = data["text"].to_list()
        texts = self.parse_str_list(texts)
        self.document_vectorizer.fit(documents=texts, vocabulary=vocabulary)
        tf_repr = self.document_vectorizer.transform(texts)
        cluwords_bow_repr = self.combine_bow_and_semantic_matrices(tf_repr, semantic_matrix)
        cluwords_textual_repr = self.document_vectorizer.map_documents_to_tokens(cluwords_bow_repr)
        data["cluwords_textual_repr"] = cluwords_textual_repr
        data["cluwords_repr"] = [cw_doc.toarray().flatten() for cw_doc in cluwords_bow_repr]
        
        return cluwords_bow_repr, data
