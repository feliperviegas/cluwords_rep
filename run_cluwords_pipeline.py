from typing import Dict
import yaml
from config.class_mapping import TASKS_MAPPING


# Read the flow configuration from a YAML file
def read_flow_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Process the flow based on the specified order
class FlowProcessor:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.flow_order = self.config['flow_order']
        self.task_objects = []  # Store instantiated task objects
        self.data_source = None

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def process_flow_step(self, step: object, config: Dict) -> object:
        if step == "Input":
            self.data_source = config['data_source']
            return None
        if step == 'Data':
            return TASKS_MAPPING[config['method']]()
        elif step == 'Embedding':
            return TASKS_MAPPING[config['method']](emb_file_source=config['data_source'], binary=config['binary'], vocabulary_file_source=config['vocabulary'])
        elif step == 'Clustering':
            return  TASKS_MAPPING[config['method']](n_threads=config['num_threads'], num_neighbors=config['num_neighbors'])
        elif step == 'Filtering':
            return TASKS_MAPPING[config['method']](threshold_value=config['threshold_value'])
        elif step == 'POSFiltering':
            return TASKS_MAPPING[config['method']](pos_filters=config['pos_filters'])
        elif step == 'Weighting':
            return TASKS_MAPPING[config['method']]()
        elif step == 'Writer':
            return TASKS_MAPPING[config['method']](cluwords_repr_path=config["cluwords_repr_path"], data_path=config["data_path"])

    def process_flow(self):
        for step in self.flow_order:
            task = self.process_flow_step(step, self.config[step.lower()])
            if task:
                self.task_objects.append(task)

    def execute_flow(self):
        for task in self.task_objects:
            if task.get_parent_class_name() == 'DataTask':
                self.data = task.execute(data_source=self.data_source)
            elif task.get_parent_class_name() == "EmbeddingTask":
                self.word_vector_space, self.vocabulary = task.execute()
            elif task.get_parent_class_name() == 'ClusteringTask':
                self.knn_labels, self.knn_distances = task.execute(word_vectors=self.word_vector_space, vocabulary=self.vocabulary)
            elif task.__class__.__name__ == 'ThresholdTask':
                self.semantic_matrix = task.execute(knn_labels=self.knn_labels, knn_distances=self.knn_distances, vocabulary=self.vocabulary)
            elif task.__class__.__name__ == 'PartOfSpeechTask':
                self.semantic_matrix = task.execute(semantic_matrix=self.semantic_matrix, vocabulary=self.vocabulary)
            elif task.get_parent_class_name() == 'WeightingTask':
                self.cluwords_bow_repr, self.data = task.execute(data=self.data, 
                                                                 vocabulary=self.vocabulary, 
                                                                 semantic_matrix=self.semantic_matrix)
            elif task.__class__.__name__ == 'WriterTask':
                task.execute(cluwords_repr=self.cluwords_bow_repr, 
                             data_df=self.data)
            else:
                print("ops")    
            
# Example usage
if __name__ == "__main__":
    config_file = 'flow_config.yaml'
    processor = FlowProcessor(config_file)
    processor.process_flow()
    processor.execute_flow()
    print("Done")