
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from sklearn.preprocessing import MinMaxScaler

from ..core.files import MlFiles
from .feature_importance_model import FeatureImportanceModel


class FeatureImportanceDropColumn(MlFiles):
    '''
    Estimate feature importance based on a model.
    How it works: Drops a single column, re-train and analyze how model performance
    is degraded (respect to validation dataset). Most important features will
    make the model perform much worse when dropped, unimportant features will
    not affect performance
    '''

    def __init__(self, model):
        super().__init__(model)

    def change_dataset(self, col):
        """ Change datasets for column 'col' """
        x_train = self.x_train.drop(c, axis=1)
        x_val = self.x_val.drop(c, axis=1)
        return x_train, self.y_train, x_val, self.y_val

    def train_and_loss(self, x_train, y_train, x_val, y_val):
        """
        Train (if necesary) and calculate loss
        In this case, there is no training, just calculate the loss
        """
        model_clone = self.model_clone(x_train, y_train, x_val, y_val)
        model_clone.model_train()
        return model_clone.model_eval_validate()
