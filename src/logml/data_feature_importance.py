#!/usr/bin/env python

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import warnings

from IPython.display import display
from scipy.cluster import hierarchy as hc

from .config import CONFIG_DATASET_FEATURE_IMPORTANCE
from .files import MlFiles


class DataFeatureImportance(MlFiles):
    '''
    Perform feature importance / feature selection analysis
    '''

    def __init__(self, datasets_df, config, set_config=True):
        super().__init__(config, CONFIG_DATASET_FEATURE_IMPORTANCE)
        self.datasets_df = datasets_df
        if set_config:
            self._set_from_config()

    def __call__(self):
        ''' Feature importance '''
        if not self.enable:
            self._info(f"Dataset feature importance / feature selection disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_FEATURE_IMPORTANCE}', enable='{self.enable}'")
            return True
        self._info("Feature importance / feature selection: Start")
        self._info("Feature importance / feature selection: End")
        return True
