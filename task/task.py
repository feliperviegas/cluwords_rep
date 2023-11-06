# Base abstract class for all tasks
from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self):
        self._parent_class_name = self.__class__.__mro__[1].__name__
        
    def get_parent_class_name(self) -> str:
        return self._parent_class_name

    @abstractmethod
    def execute(self):
        pass
    
    