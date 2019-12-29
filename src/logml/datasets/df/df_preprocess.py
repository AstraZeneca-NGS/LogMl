#!/usr/bin/env python

import numpy as np
import pandas as pd
import re

from collections import namedtuple
from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog
from .df_normalize import DfNormalize
from .df_impute import DfImpute


class DfPreprocess(MlLog):
    '''
    DataFrame preprocessing: impute missing data, normalize
    '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_PREPROCESS)
        self.config = config
        self.df = df
        self.outputs = outputs
        self.model_type = model_type
        self.balance = False
        self.normalize_df = None
        if set_config:
            self._set_from_config()

    def _balance(self):
        """ Balance an unbalanced dataset """
        if not self.balance:
            self._debug(f"Balance: Disabled, skipping")
            return True
        if self.model_type != 'classification':
            self._debug(f"Balance: Cannot balance for model type '{self.model_type}', skipping")
            return False
        y = self.df[self.outputs]
        uniq, counts = np.unique(y, return_counts=True)
        self._debug(f"Balance: Start, dataframe shape: {self.df.shape}, percents: {counts / counts.sum()}, counts: {counts}, cathegories: {uniq}")
        counts_max = counts.max()
        for u, c in zip(uniq, counts):
            y = self.df[self.outputs]
            num = counts_max - c
            self._debug(f"Balance: Adding samples for outputs={u}, count={c}, to add: {num}, dataframe shape: {self.df.shape}")
            if num == 0:
                continue
            replace = num > c
            weights = (y == u).astype('float').to_numpy().flatten()
            rows_to_add = self.df.sample(n=num, replace=replace, weights=weights)
            self.df = pd.concat([self.df, rows_to_add])
        y = self.df[self.outputs]
        uniq, counts = np.unique(y, return_counts=True)
        self._debug(f"Balance: End, dataframe shape: {self.df.shape}, percents: {counts / counts.sum()}, counts: {counts}, cathegories: {uniq}")

    def __call__(self):
        """
        Preprocess dataframe columns
        Returns a new (transformed) dataset
        """
        if not self.enable:
            self._debug(f"Preprocessing dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', enable='{self.enable}'")
            return self.df
        self._debug("Preprocessing dataframe: Start")
        self.impute_df = DfImpute(self.df, self.config, self.outputs, self.model_type)
        self.df = self.impute_df()
        self.normalize_df = DfNormalize(self.df, self.config, self.outputs, self.model_type)
        self.df = self.normalize_df()
        self._balance()
        self._debug("Preprocessing dataframe: End")
        return self.df
