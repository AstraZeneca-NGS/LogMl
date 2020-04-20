
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traceback

from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles
from .feature_importance_model import FeatureImportanceModel


class FeatureImportancePermutation(FeatureImportanceModel):
    '''
    Estimate feature importance based on a model.
    How it works: Suffle a column and analyze how model performance is
    degraded. Most important features will make the model perform much
    worse when shuffled, unimportant features will not affect performance

    To estimate a p-value, it uses a ranked test by comparing to resutls from
    randomly shuffled columns
    '''

    def __init__(self, model, model_name, rand_columns, num_iterations):
        super().__init__(model, model_name, rand_columns, num_iterations)
        self.importance_name = 'permutation'

    def dataset_change(self, col_name):
        """ Change datasets for column 'col_name' """
        col_ori = self.datasets.shuffle_input(col_name)
        return col_ori

    def dataset_restore(self, col_name, col_ori):
        """ Restore column 'col_name' using values 'col_ori' """
        self.datasets.shuffle_input(col_name, col_ori)

    def initialize(self):
        """ Initialzie the model (the model is trained only once) """
        self._debug(f"Feature importance ({self.importance_name}, {self.model_type}): Initialize. Model fit")
        self.model.model_train()

    def loss(self, is_base=False):
        """
        Train (if necesary) and calculate loss
        In this case, there is no training, just evaluate the loss
        """
        self.model.model_eval_validate()
        return self.model.eval_validate_values if self.is_cv else self.model.eval_validate
