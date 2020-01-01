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
        self.pca_augment = None
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
        cols = "', '".join([c for c in self.df.columns])
        self._debug(f"Augment dataframe: Start. Fields ({len(self.df.columns)}): ['{cols}']")
        self._pca()
        cols = "', '".join([c for c in self.df.columns])
        self._debug(f"Augment dataframe: End. Fields ({len(self.df.columns)}): ['{cols}']")
        return self.df

    def _pca(self):
        self.pca_augment = DfAugmentPca(self.df, self.config, self.outputs, self.model_type)
        ret = self.pca_augment()
        if ret is None:
            self._debug("Augment dataframe: Could not do PCA")
            return False
        else:
            self.df = self.df.join(ret)
            return True


class DfAugmentPca(CountAndFields):
    ''' Augment dataset by adding principal components '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, 'pca', df.columns, outputs)
        self.sk_pca_by_name = dict()

    def calc(self, nf, x):
        """Calculate 'num' PCAs using 'fields' from dataframe
        Returns: A dataframe of PCAs (None on failure)
        """
        self._debug(f"Calculating PCA: Start, name={nf.name}, num={nf.number}, fields:{nf.fields}")
        pca = PCA(n_components=nf.number)
        pca.fit(x)
        self.sk_pca_by_name[nf.name] = pca
        xpca = pca.transform(x)
        self._debug(f"Calculating PCA: End")
        return xpca
