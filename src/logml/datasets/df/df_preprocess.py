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
        self.df = df
        self.config = config
        self.outputs = outputs
        self.model_type = model_type
        self.normalize_df = None
        if set_config:
            self._set_from_config()

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
        self._debug("Preprocessing dataframe: End")
        return self.df
