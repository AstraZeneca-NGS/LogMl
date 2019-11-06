
import math
import numpy as np
import matplotlib.pyplot as plt
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
    '''

    def __init__(self, model):
        super().__init__(model)

    def change_dataset(self, col):
        """ Change datasets for column 'col' """
        x_val = self.x_val.copy()
        xi = np.random.permutation(x_val[col])
        x_val[col] = xi
        return None, None, x_val, self.y_val

    def initialize(self):
        """ Initialzie the model (the model is trained only once) """
        self.model.fit(self.x_train, self.y_train)

    def loss(self, x_train, y_train, x_val, y_val):
        """
        Train (if necesary) and calculate loss
        In this case, there is no training, just evaluate the loss
        """
        self.model_set_datasets(self.model, x_train, y_train, x_val, y_val)
        self.model.model_eval_validate()
        return self.model.eval_validate
