
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles
from .feature_importance_model import FeatureImportanceModel


class FeatureImportanceDropColumn(FeatureImportanceModel):
    '''
    Estimate feature importance based on a model.
    How it works: Drops a single column, re-train and analyze how model performance
    is degraded (respect to validation dataset). Most important features will
    make the model perform much worse when dropped, unimportant features will
    not affect performance
    '''

    def __init__(self, model, model_name):
        super().__init__(model, model_name)
        self.importance_name = 'drop column'

    def dataset_change(self, col_name):
        """ Change datasets for column 'col_name' """
        col_ori = self.datasets.zero_input(col_name)
        return col_ori

    def dataset_restore(self, col_name, col_ori):
        """ Restore column 'col_name' using values 'col_ori' """
        self.datasets.zero_input(col_name, col_ori)

    def loss(self):
        """ Train and calculate loss """
        model_clone = self.model.clone()
        model_clone.model_train()
        model_clone.model_eval_validate()
        return model_clone.eval_validate
        return model_clone.eval_validate_values if self.is_cv else model_clone.eval_validate
