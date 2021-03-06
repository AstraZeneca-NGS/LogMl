#!/usr/bin/env python

import numpy as np
import pandas as pd
import re
from collections import namedtuple

from ...core import MODEL_TYPE_CLASSIFICATION
from ...core.config import CONFIG_DATASET_PREPROCESS
from ...core.log import MlLog
from .methods_fields import MethodsFields

# Method names
NORMALIZATION_METHODS = ['log', 'log_standard', 'log1p', 'log1p_standard', 'maxabs', 'minmax', 'minmax_neg', 'quantile', 'skip', 'standard']


class DfNormalize(MethodsFields):
    """
    DataFrame normalization: Normalize / re-scale inputs
    """

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_PREPROCESS, 'normalize', NORMALIZATION_METHODS, df.columns, outputs)
        self.df = df
        self.model_type = model_type
        self.outputs = set(outputs)
        # if set_config:
        #     self._set_from_config()
        self._initialize()

    def __call__(self):
        """
        Normalize inputs
        Returns a new (normalized) dataset
        """
        if not self.enable:
            self._debug(f"Normalizing dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_PREPROCESS}', sub-section 'normalize', enable='{self.enable}'")
            return self.df
        self._debug("Normalizing dataframe: Start")
        self._normalize()
        self._debug("Normalizing dataframe: End")
        return self.df

    def is_numertic(self, col_name):
        return np.issubdtype(self.df[col_name].dtype, np.number)

    def is_range_01(self, col_name):
        """
        Is this variable in [0,1] range?
        We say the varaible is in [0.0, 1.0] range if the min is
        in [0.0 , 0.1] and the max is in [0.9, 1.0]
        """
        xi = self.df[col_name]
        xi_min = xi.unique().min()
        xi_max = xi.unique().max()
        return 0.0 <= xi_min and xi_min <= 0.1 and 0.9 <= xi_max and xi_max <= 1.0

    def _normalize(self):
        """ Normalize variables """
        self._debug("Normalizing dataset (dataframe): Start")
        fields_to_normalize = list(self.df.columns)
        is_classification = (self.model_type == MODEL_TYPE_CLASSIFICATION)
        for c in fields_to_normalize:
            if not self.is_numertic(c):
                self._debug(f"Normalize variable '{c}' is not numeric, skipping")
                continue
            if self.is_range_01(c):
                self._debug(f"Normalize variable '{c}' is in [0, 1], skipping")
                continue
            if is_classification and c in self.outputs:
                self._debug(f"Normalize variable '{c}' is an output varaible for a classification model, skipping")
                continue
            nm = self.find_method(c)
            xi = self.df[c]
            self._debug(f"Before normalization '{c}': mean={np.nanmean(xi)}, std={np.nanstd(xi)}, range=[{xi.min()}, {xi.max()}]")
            if nm is None:
                self._debug(f"Normalize field '{c}': No method defined, skipping")
            else:
                self._info(f"Normalizing field '{c}', method {nm.__name__}")
                # Normalize and replace column
                xi = nm(xi)
                self._debug(f"After normalization '{c}': mean={np.nanmean(xi)}, std={np.nanstd(xi)}, range=[{xi.min()}, {xi.max()}]")
                self.df[c] = xi
        self._debug("Normalizing dataset (dataframe): End")

    def _normalize_log(self, xi):
        """ Normalize using 'log' method """
        return np.log(xi)

    def _normalize_log_standard(self, xi):
        """ Normalize using 'log' method """
        return self._normalize_standard(np.log(xi))

    def _normalize_log1p(self, xi):
        """ Normalize using 'log1p' method """
        return np.log1p(xi)

    def _normalize_log1p_standard(self, xi):
        """ Normalize standard normalization on log(x + 1) """
        return self._normalize_standard(np.log1p(xi))

    def _normalize_maxabs(self, xi):
        """ Normalize using 'maxabs' method """
        return xi / np.max(np.abs(xi))

    def _normalize_minmax(self, xi):
        """ Normalize using 'minmax' method. Convert to interval [0, 1] """
        mi = np.min(xi)
        ma = np.max(xi)
        if (ma - mi) <= 0.0:
            return xi
        return (xi - mi) / (ma - mi)

    def _normalize_minmax_neg(self, xi):
        """ Normalize using 'minmax' method. Convert to interval [-1, 1] """
        mi = np.min(xi)
        ma = np.max(xi)
        if (ma - mi) <= 0.0:
            return xi
        return 2.0 * (xi - mi) / (ma - mi) - 1.0

    def _normalize_quantile(self, xi):
        """ Normalize using 'quantile' method """
        return None

    def _normalize_skip(self, xi):
        """ Skip normalization: Normalize using 'skip' method (i.e. do not normalize) """
        return xi

    def _normalize_standard(self, xi):
        """ Normalize using 'standard' method (substract the mean and divide by std)"""
        me = np.nanmean(xi)
        std = np.nanstd(xi)
        if std <= 0.0:
            return (xi - me)
        return (xi - me) / std
