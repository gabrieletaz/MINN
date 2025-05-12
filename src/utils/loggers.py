from typing import List
from clearml import Task
import pandas as pd

from src.utils.utils import Prittifier

class DataFrameLogger:
    def __init__(self, clearml_task: Task) -> None:
        self._results = []
        self._task = clearml_task
        
    def log_results(self, results:dict):
        self._results.append(results)
        
    def get_results(self) -> pd.DataFrame:
        if self._results:
            return pd.DataFrame(self._results)
        else:
            return pd.DataFrame()
    
    def clear_results(self):
        self._results = []
    
    def report_as_table(self, title:str, iteration:int = 0):
        assert self._task is not None, "Task is not set"
        self._task.get_logger().report_table(title=title, series=title, iteration=iteration, table_plot=Prittifier().prittify_column_names(self.get_results()))
        
    def report_as_artifact(self, name):
        assert self._task is not None, "Task is not set"
        self._task.upload_artifact(name, self.get_results())