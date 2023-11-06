import yaml
from clustering_task.clustering_task import ClusteringTask
from data_task.data_task import DataTask
from filtering_task.filtering_task import FilteringTask

from task.task import Task
from weighting_task.weighting_task import WeightingTask


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

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def process_flow_step(self, step, config):
        if step == 'Data':
            return DataTask()
        elif step == 'Clustering':
            return ClusteringTask(config['num_neighbors'])
        elif step == 'Filtering':
            return FilteringTask(config['threshold_value'])
        elif step == 'Weighting':
            return WeightingTask()

    def process_flow(self):
        for step in self.flow_order:
            task = self.process_flow_step(step, self.config[step.lower()])
            self.task_objects.append(task)

    def execute_flow(self, data_source):
        for task in self.task_objects:
            if task.__class__.__name__ == 'DataTask':
                self.data = task.execute(data_source=data_source)
            elif task.__class__.__name__ == 'ClusteringTask':
                self.semantic_matrix = task.execute(self.data)
            elif task.__class__.__name__ == 'FilteringTask':
                self.semantic_matrix = task.execute(self.semantic_matrix)
            elif task.__class__.__name__ == 'WeightingTask':
                self.result = task.execute(self.data, self.semantic_matrix)
                print(self.result)
            
# Example usage
if __name__ == "__main__":
    config_file = 'flow_config.yaml'
    processor = FlowProcessor(config_file)
    processor.process_flow()
    # processor.execute_flow()
    print("Done")