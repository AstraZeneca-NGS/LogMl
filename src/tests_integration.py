
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import os
import random
import sys
import time
import unittest

from logml.core.config import Config, CONFIG_CROSS_VALIDATION, CONFIG_DATASET, CONFIG_HYPER_PARAMETER_OPTMIMIZATION, CONFIG_LOGGER, CONFIG_MODEL
from logml.datasets import Datasets, DatasetsDf
from logml.core.files import MlFiles
from logml.core.log import MlLog
from logml.core import LogMl
from logml.models import Model
from logml.core.registry import MlRegistry, DATASET_AUGMENT, DATASET_CREATE, DATASET_INOUT, DATASET_PREPROCESS, DATASET_SPLIT, MODEL_CREATE, MODEL_EVALUATE, MODEL_TRAIN

from logml.core.scatter_gather import ScatterGather

DEBUG = os.getenv('TEST_INTEGRATION_DEBUG', False)


class TestLogMlIntegration(unittest.TestCase):

    def setUp(self):
        MlLog().set_log_level(logging.CRITICAL)
        if DEBUG:
            MlLog().set_log_level(logging.DEBUG)
        MlRegistry().reset()

    def test_linear3(self):
        """ Simple linear model (without any noise) """
        config_file = os.path.join('tests', 'integration', 'config', 'linear3.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        ret = config()
        ml = LogMl(config=config, debug=DEBUG)
        ml()
        # Check data preprocessing
        dp = ml.datasets.dataset_preprocess
        df = ml.datasets.dataset
        epsilon = 0.001
        for xi in ['x1', 'x2', 'x3']:
            self.assertTrue(abs(df[xi].mean()) < epsilon)
            self.assertTrue(abs(df[xi].std() - 1) < epsilon)
        # Check feature feature importance
        fidf = ml.dataset_feature_importance.results.df
        self.assertEqual('x1', fidf.index[0])
        self.assertEqual('x2', fidf.index[1])
        self.assertEqual('x3', fidf.index[2])
        # Check model search results
        mrdf = ml.model_results.df
        modsearch_best = mrdf.index[0]
        modsearch_first = mrdf.iloc[0]
        self.assertTrue(modsearch_best.startswith("sklearn.linear_model.LinearRegression"))
        self.assertEqual(modsearch_first.train, 0.0)
        self.assertEqual(modsearch_first.validation, 0.0)

    def test_linear3_with_scatter_and_gather(self):
        # run with scatter mode 'pre'
        os.system('rm -rvf data/linear3/linear3.pkl data/linear3/model')
        config_file = os.path.join('tests', 'integration', 'config', 'linear3.yaml')
        config = Config(argv=['logml.py', '-c', config_file, '-s', '4', '-n', 'pre'])
        config()
        ml = LogMl(config=config)
        ml()

        # run with scatter number 4
        for scatter_numb in range(4):
            os.system('rm -rvf data/linear3/linear3.pkl data/linear3/model')
            config = Config(argv=['logml.py', '-c', config_file, '-s', '4', '-n', str(scatter_numb)])
            config()
            ml = LogMl(config=config)
            ml()

        # run with scatter mode 'gather'
        os.system('rm -rvf data/linear3/linear3.pkl data/linear3/model')
        config = Config(argv=['logml.py', '-c', config_file, '-s', '4', '-n', 'gather'])
        config()
        ml = LogMl(config=config)
        ml()

        # Check feature feature importance
        fidf = ml.dataset_feature_importance.results.df
        # breakpoint()
        self.assertEqual('x1', fidf.index[0])
        self.assertEqual('x2', fidf.index[1])
        self.assertEqual('x3', fidf.index[2])
        # Check model search results
        mrdf = ml.model_results.df
        modsearch_best = mrdf.index[0]
        modsearch_first = mrdf.iloc[0]
        self.assertTrue(modsearch_best.startswith("sklearn.linear_model.LinearRegression"))
        self.assertEqual(modsearch_first.train, 0.0)
        self.assertEqual(modsearch_first.validation, 0.0)

        detailed_table_results_by_model = ml.dataset_feature_importance.detailed_table_results_by_model
        self.assertTrue('RandomForest' in detailed_table_results_by_model.keys())
        self.assertTrue('ExtraTrees' in detailed_table_results_by_model.keys())
        self.assertTrue('GradientBoosting' in detailed_table_results_by_model.keys())
        self.assertTrue('importance_permutation' in detailed_table_results_by_model['RandomForest'][0])
        self.assertTrue('importance_dropcol' in detailed_table_results_by_model['RandomForest'][1])
        self.assertTrue('importance_permutation' in detailed_table_results_by_model['ExtraTrees'][0])
        self.assertTrue('importance_dropcol' in detailed_table_results_by_model['ExtraTrees'][1])
        self.assertTrue('importance_permutation' in detailed_table_results_by_model['GradientBoosting'][0])
        self.assertTrue('importance_dropcol' in detailed_table_results_by_model['GradientBoosting'][1])

        # remove the scatter folder
        scatter_path = Path('.') / f"scatter_{ml.config.scatter_total}_{ml.config.config_hash}"
        ScatterGather.remove_scatter_folder(scatter_path)

    def test_linear3c(self):
        """ Linear model (with noise and missing values) """
        config_file = os.path.join('tests', 'integration', 'config', 'linear3c.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        ret = config()
        ml = LogMl(config=config, debug=DEBUG)
        ml()
        # Check results
        dp = ml.datasets.dataset_preprocess
        df = ml.datasets.dataset
        # Check data preprocess: Remove 'missing output rows' (1% of rows removed)
        self.assertTrue(df.shape[0] < 1990)
        # Check data preprocess: Convert to one hot
        for c in ['c1:high', 'c1:mid', 'c1:low', 'c2:very_high', 'c2:high', 'c2:mid', 'c2:low', 'c2:very_low']:
            self.assertTrue(c in df.columns, f"Missing one-hot column {c}, {df.columns}")
        # Check data preprocess: Add 'na' columns
        for c in ['x1_na', 'x2_na', 'x3_na']:
            self.assertTrue(c in df.columns, f"Missing '*_na' column {c}: {df.columns}")
        # Check data preprocessing: Normalization
        epsilon = 0.001
        for xi in ['x1', 'x2', 'x3']:
            self.assertTrue(abs(df[xi].mean()) < epsilon)
            self.assertTrue(abs(df[xi].std() - 1) < epsilon)
        # Check feature feature importance
        fidf = ml.dataset_feature_importance.results.df
        print(fidf)
        self.assertEqual('x1', fidf.index[0])
        self.assertEqual('x2', fidf.index[1])
        self.assertEqual('x3', fidf.index[2])
        # Check model search results
        mrdf = ml.model_results.df
        modsearch_best = mrdf.index[0]
        modsearch_first = mrdf.iloc[0]
        self.assertTrue(modsearch_best.startswith("sklearn.linear_model.LinearRegression"))
        self.assertTrue(modsearch_first.train < 0.1)
        self.assertTrue(modsearch_first.validation < 0.1)

    def test_class3(self):
        """ Classification problem (with noise and missing values) """
        config_file = os.path.join('tests', 'integration', 'config', 'class3.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        ret = config()
        ml = LogMl(config=config, debug=DEBUG)
        ml()
        # Check results
        dp = ml.datasets.dataset_preprocess
        df = ml.datasets.dataset
        # Check data preprocess: Remove 'missing output rows' (1% of rows removed)
        self.assertTrue(df.shape[0] < 1990)
        # Check data preprocess: Add 'na' columns
        for c in ['x1_na', 'x2_na', 'x3_na']:
            self.assertTrue(c in df.columns, f"Missing '*_na' column {c}: {df.columns}")
        # Check data preprocessing: Normalization
        epsilon = 0.001
        for xi in ['x1', 'x2', 'x3']:
            self.assertTrue(abs(df[xi].mean()) < epsilon)
            self.assertTrue(abs(df[xi].std() - 1) < epsilon)
        # Check feature feature importance
        fidf = ml.dataset_feature_importance.results.df
        print(fidf)
        self.assertEqual('x1', fidf.index[0])
        self.assertEqual('x2', fidf.index[1])
        # Check model search results
        mrdf = ml.model_results.df
        modsearch_best = mrdf.index[0]
        modsearch_first = mrdf.iloc[0]
        self.assertTrue(modsearch_best.startswith("sklearn.linear_model.LogisticRegressionCV"))
        self.assertTrue(modsearch_first.train < 0.1)
        self.assertTrue(modsearch_first.validation < 0.1)

    # def test_example1(self):
    #     pass
    #
    # def test_example2(self):
    #     pass
    #
    # def test_example3(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
