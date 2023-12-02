from clustering_task.clustering_task import ClusteringTask
from clustering_task.hnsw_task import HnswTask
from data_task.data_task import DataTask
from data_task.pandas_task import PandasTask
from embedding_task.embedding_task import EmbeddingTask
from embedding_task.gensim_emb_task import GensimEmbeddingTask
from filtering_task.pos_task import PartOfSpeechTask
from filtering_task.threshold_task import ThresholdTask
from weighting_task.tf_task import TfTask
from writer.writer_task import WriterTask


TASKS_MAPPING = {
    "DataTask": DataTask,
    "PandasTask": PandasTask,
    "EmbeddingTask": EmbeddingTask,
    "GensimEmbeddingTask": GensimEmbeddingTask,
    "ClusteringTask": ClusteringTask,
    "HnswTask": HnswTask,
    "ThresholdTask": ThresholdTask,
    "PartOfSpeechTask": PartOfSpeechTask,
    "TfTask": TfTask,
    "WriterTask": WriterTask,
}