#!/usr/bin/env python

import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.decomposition import PCA

from ...core.config import CONFIG_DATASET_AUGMENT
from ...core.log import MlLog
from .df_normalize import DfNormalize
from .df_impute import DfImpute
from .methods_fields import CountAndFields


class DfAugment(MlLog):
    '''
    DataFrame augmentation
    '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_AUGMENT)
        self.config = config
        self.df = df
        self.outputs = outputs
        self.model_type = model_type
        self.pca = dict()
        if set_config:
            self._set_from_config()

    def __call__(self):
        """
        Augment dataframe
        Returns a new (augmented) dataset
        """
        if not self.enable:
            self._debug(f"Augment dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_AUGMENT}', enable='{self.enable}'")
            return self.df
        self._debug("Augment dataframe: Start")
        self._pca()
        self._debug("Augment dataframe: End")
        return self.df

    def _pca(self):
        pca = DfAugmentPca(self.df, self.config, self.outputs, self.model_type)
        ret, self.pca_transform = pca()
        if ret is None:
            self._debug("Augment dataframe: Could not do PCA")
            return False
        else:
            self.df = pd.join([self.df, ret])
            return True


class DfAugmentPca(CountAndFields):
    ''' Augment dataset by adding principal components '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, 'pca', df.columns, outputs)
        self.sk_pcas = list()

    def calc(self, num, fields, x):
        """Calculate 'num' PCAs using 'fields' from dataframe
        Returns: A dataframe of PCAs (None on failure)
        """
        self._debug(f"Calculating PCA: Start, num={num}, fields:{fields}")
        pca = PCA(n_components=num)
        pca.fit(x)
        self.sk_pcas.append(pca)
        dout = pca.transform(x)
        self._debug(f"Calculating PCA: End")
        return dout
