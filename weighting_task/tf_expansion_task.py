from typing import List
from spacy.tokens.token import Token
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from itertools import product
from utils.document_vectorizer import DocumentVectorizer
from weighting_task.weighting_task import WeightingTask


class TfExpansionTask(WeightingTask):
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
    
    @staticmethod
    def get_expansions(text: List[str], semantic_matrix: csr_matrix, index_vocab: dict, inverted_index_vocab: dict) -> List[dict]:
        text_struct = []
        for idx in range(len(text)):
            word = text[idx].text
            if word in index_vocab:
                index_x = index_vocab[word]
                index_y = semantic_matrix[index_vocab[word]].nonzero()[1]
                # Remove root word
                index_y = index_y[index_y != index_x]
                if len(index_y) > 0:
                    neighbor_words = [inverted_index_vocab[elem] for elem in index_y]
                    text_struct.append([word] + neighbor_words)
                else:
                    text_struct.append([word])
            else:
                text_struct.append([word])
        
        return text_struct
    
    @staticmethod
    def chunks(input_list: List, n: int) -> List:
        """
        Create n-sized chunks from input list.
        """
        output_list = []
        for i in range(0, len(input_list), n):
            output_list.append(input_list[i:i + n])
        
        return output_list
        
        
    def execute(self, data: pd.DataFrame, vocabulary: List[str], semantic_matrix: csr_matrix) -> np.array:
        texts = data["text"].to_list()
        index_vocab = {word:num for num, word in enumerate(vocabulary)}
        inverted_index_vocab = {num: word for num, word in enumerate(vocabulary)}
        texts_struct = []
        for text in texts:
            texts_struct.append(self.get_expansions(text, semantic_matrix, index_vocab, inverted_index_vocab))
        
        chunks_texts_struct = self.chunks(texts_struct, 10)
        
        expanded_texts = []
        for chunk_text in chunks_texts_struct:
            for text in chunk_text:
                # Use the product function to generate all combinations
                combinations = list(product(*text))
        
                # Combine elements and create the desired strings
                expanded_texts.append([' '.join(combination) for combination in combinations])
        
        texts = self.parse_str_list(texts)
        self.document_vectorizer.fit(documents=texts, vocabulary=vocabulary)
        tf_repr = self.document_vectorizer.transform(texts)
        cluwords_bow_repr = self.combine_bow_and_semantic_matrices(tf_repr, semantic_matrix)
        # cluwords_textual_repr = self.document_vectorizer.map_documents_to_tokens(cluwords_bow_repr)
        # data["cluwords_textual_repr"] = cluwords_textual_repr
        data["cluwords_repr"] = [cw_doc.toarray().flatten() for cw_doc in cluwords_bow_repr]
        cluwords_textual_repr = pd.DataFrame([{"cluwords_textual_repr": texts} for texts in expanded_texts])
        
        data = pd.concat([data , cluwords_textual_repr ], axis=1)
        
        return cluwords_bow_repr, data
