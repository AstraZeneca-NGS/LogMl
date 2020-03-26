
import logging
import numpy as np
import pandas as pd
import os
import random
import sys
import time
import unittest

from logml.core import LogMl
from logml.core.config import Config, CONFIG_CROSS_VALIDATION, CONFIG_DATASET, CONFIG_HYPER_PARAMETER_OPTMIMIZATION, CONFIG_LOGGER, CONFIG_MODEL
from logml.core.files import set_plots, MlFiles
from logml.core.log import MlLog
from logml.core.registry import MlRegistry, DATASET_AUGMENT, DATASET_CREATE, DATASET_INOUT, DATASET_PREPROCESS, DATASET_SPLIT, MODEL_CREATE, MODEL_EVALUATE, MODEL_PREDICT, MODEL_TRAIN
from logml.datasets import Datasets, DatasetsDf, DatasetsCv, DatasetsDf
from logml.feature_importance.data_feature_importance import DataFeatureImportance
from logml.feature_importance.pvalue_fdr import LogisticRegressionWilks, MultipleLogisticRegressionWilks, PvalueLinear
from logml.models import Model

DEBUG = os.getenv('TEST_UNIT_DEBUG', False)
# DEBUG = True


def array_equal(a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a1_nan_idx = np.isnan(a1)
    a2_nan_idx = np.isnan(a2)
    if not np.array_equal(a1_nan_idx, a2_nan_idx):
        return False
    return np.array_equal(a1[~a1_nan_idx], a2[~a2_nan_idx])


def is_close(x, y, epsilon=0.000001):
    return abs(x - y) < epsilon


def is_sorted(x):
    """ Is numpy array 'x' sorted? """
    return np.all(x[:-1] <= x[1:])


def rm(file):
    ''' Delete file, if it exists '''
    if os.path.exists(file):
        os.remove(file)


class TestLogMl(unittest.TestCase):

    def setUp(self):
        MlLog().set_log_level(logging.CRITICAL)
        if DEBUG:
            MlLog().set_log_level(logging.DEBUG)
        set_plots(disable=True, show=False, save=False)
        MlRegistry().reset()

    def test_config_001(self):
        ''' Test objects parameters from config and file names '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_config_001.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        ret = config()
        mldataset = Datasets(config)
        logml = LogMl(config=config)
        mltrain = Model(config, mldataset)
        self.assertTrue(ret)
        self.assertEqual(mltrain.model_name, 'model_001')
        self.assertEqual(mldataset.get_file_name(), os.path.join('tests', 'tmp', 'test_001.pkl'))
        self.assertEqual(mltrain.get_file_name(), os.path.join('tests', 'tmp', 'model', f"model_001.{mltrain._id}.pkl"))
        self.assertEqual(logml.hyper_parameter_optimization.enable, False)
        self.assertEqual(config.get_parameters_functions('dataset_augment'), {'num_augment': 10})

    # TODO: Obsolete test. Replace
    def test_config_002(self):
        pass

    def test_config_003(self):
        config = Config(os.path.join('tests', 'unit', 'config', 'test_config_003.yaml'), argv=list())
        ret = config()
        logml = LogMl(config=config)
        hopt = logml.hyper_parameter_optimization
        self.assertEqual(hopt.algorithm, 'tpe')
        self.assertEqual(hopt.max_evals, 123)
        self.assertEqual(hopt.show_progressbar, True)
        self.assertEqual(hopt.enable, True)

    def test_dataset_001(self):
        def test_dataset_001_augment(dataset, num_augment=1):
            assert num_augment == 10

        def test_dataset_001_create(num_create):
            assert num_create == 42
            return [1, 2, 3, 4]

        MlRegistry().register(DATASET_CREATE, test_dataset_001_create)
        MlRegistry().register(DATASET_AUGMENT, test_dataset_001_augment)
        config = Config(os.path.join('tests', 'unit', 'config', 'test_dataset_001.yaml'), argv=list())
        config()
        self.assertEqual(config.get_parameters_functions('dataset_augment'), {'num_augment': 10})
        self.assertEqual(config.get_parameters_functions('dataset_create'), {'num_create': 42})
        ds = Datasets(config)
        ret = ds.create()
        self.assertTrue(ret)
        self.assertEqual(ds.dataset, [1, 2, 3, 4])
        ret = ds.augment()
        self.assertTrue(ret)

    def test_dataset_002(self):
        ''' Check Datasets.__call__() '''
        def test_dataset_002_augment(dataset, num_augment=1):
            assert num_augment == 10
            dataset.append(5)
            dataset.append(6)
            return dataset

        def test_dataset_002_create(num_create):
            assert num_create == 42
            return [1, 2, 3, 4]

        def test_dataset_002_preprocess(dataset, add_offset):
            assert add_offset == 7
            return [add_offset + n for n in dataset]

        def test_dataset_002_split(dataset, split_validate, split_test):
            assert split_test == 0.2
            assert split_validate == 0.1
            return [1, 2, 3], [4, 5], [6]

        def test_dataset_002_inout(dataset):
            return dataset, dataset

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_dataset_002_create)
        MlRegistry().register(DATASET_INOUT, test_dataset_002_inout)
        MlRegistry().register(DATASET_AUGMENT, test_dataset_002_augment)
        MlRegistry().register(DATASET_PREPROCESS, test_dataset_002_preprocess)
        MlRegistry().register(DATASET_SPLIT, test_dataset_002_split)
        # Read config
        config = Config(os.path.join('tests', 'unit', 'config', 'test_dataset_002.yaml'), argv=list())
        config()
        # Create dataset
        ds = Datasets(config)
        # Cleanup old files
        rm(ds.get_file_name())
        ret = ds()
        # Check values
        self.assertTrue(ret)
        self.assertEqual(ds.operations_done, set([DATASET_AUGMENT, DATASET_PREPROCESS, DATASET_SPLIT]))
        self.assertEqual(ds.dataset, [8, 9, 10, 11, 5, 6])
        self.assertEqual(ds.get_train(), [1, 2, 3])
        self.assertEqual(ds.get_validate(), [4, 5])
        self.assertEqual(ds.get_test(), [6])
        # Load from pickle and check values
        ds2 = Datasets(config)
        ret = ds2.load()
        self.assertTrue(ret)
        self.assertEqual(ds.dataset, [8, 9, 10, 11, 5, 6])
        self.assertEqual(ds.dataset_train, [1, 2, 3])
        self.assertEqual(ds.dataset_validate, [4, 5])
        self.assertEqual(ds.dataset_test, [6])
        self.assertEqual(ds.operations_done, set([DATASET_AUGMENT, DATASET_PREPROCESS, DATASET_SPLIT]))

    def test_dataset_003(self):
        ''' Check Datasets.__call__(), with 'enable=False', in this case 'dataset_split' is false '''
        def test_dataset_003_augment(dataset, num_augment=1):
            assert num_augment == 10
            dataset.append(5)
            dataset.append(6)
            return dataset

        def test_dataset_003_create(num_create):
            assert num_create == 42
            return [1, 2, 3, 4]

        def test_dataset_003_split(dataset, split_test, split_validate):
            assert split_test == 0.2
            assert split_validate == 0.1
            return [1, 2, 3], [4, 5], [6]

        def test_dataset_003_inout(dataset):
            return dataset, dataset

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_dataset_003_create)
        MlRegistry().register(DATASET_INOUT, test_dataset_003_inout)
        MlRegistry().register(DATASET_AUGMENT, test_dataset_003_augment)
        MlRegistry().register(DATASET_SPLIT, test_dataset_003_split)
        # Read config
        config = Config(os.path.join('tests', 'unit', 'config', 'test_dataset_003.yaml'), argv=list())
        config()
        # Create dataset
        ds = Datasets(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        self.assertEqual(ds.dataset, [1, 2, 3, 4, 5, 6])
        self.assertEqual(ds.operations_done, set([DATASET_AUGMENT]))
        self.assertEqual(ds.dataset_xy.x, [1, 2, 3, 4, 5, 6])
        self.assertEqual(ds.dataset_xy.y, [1, 2, 3, 4, 5, 6])

    def test_dataset_004(self):
        ''' Check Datasets.__call__(), with default split method '''
        def test_dataset_004_create(num_create):
            assert num_create == 42
            return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_dataset_004_create)
        # Read config
        config = Config(os.path.join('tests', 'unit', 'config', 'test_dataset_004.yaml'), argv=list())
        config()
        # Create dataset
        ds = Datasets(config)
        rm(ds.get_file_name())
        random.seed(20190705)
        ret = ds()
        dataset_expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        dataset_expected_train = np.array([2, 5, 6, 9])
        dataset_expected_val = np.array([1, 4, 8])
        dataset_expected_test = np.array([3, 7, 10])
        self.assertTrue(ret)
        self.assertTrue(np.array_equal(ds.dataset, dataset_expected))
        self.assertTrue(np.array_equal(ds.operations_done, set([DATASET_SPLIT])))
        self.assertTrue(np.array_equal(ds.get_train(), dataset_expected_train))
        self.assertTrue(np.array_equal(ds.get_validate(), dataset_expected_val))
        self.assertTrue(np.array_equal(ds.get_test(), dataset_expected_test))
        self.assertTrue(np.array_equal(ds.get_train_xy().x, dataset_expected_train))
        self.assertTrue(np.array_equal(ds.get_validate_xy().x, dataset_expected_val))
        self.assertTrue(np.array_equal(ds.get_test_xy().x, dataset_expected_test))

    def test_dataset_005(self):
        " Test split_idx method "
        config = Config(os.path.join('tests', 'unit', 'config', 'test_dataset_005.yaml'), argv=list())
        config()
        ds = Datasets(config)
        ds.dataset = np.array([i + 100 for i in range(10)])
        ret = ds.split_idx([0, 1, 2, 3, 4], [5, 6, 7], [8, 9])
        self.assertTrue(ret)
        self.assertTrue(all(ds.dataset == np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])))
        self.assertTrue(all(ds.get_train() == np.array([100, 101, 102, 103, 104])))
        self.assertTrue(all(ds.get_validate() == np.array([105, 106, 107])))
        self.assertTrue(all(ds.get_test() == np.array([108, 109])))

    def test_dataset_006(self):
        ''' DatasetsDf test (load dataframe) and expand date/time columns and '_na' columns '''
        # Create dataset
        config = Config(os.path.join('tests', 'unit', 'config', 'test_dataset_006.yaml'), argv=list())
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check categories
        self.assertTrue(all(ds.dataset['UsageBand'].head(10) == np.array([2, 2, 2, 1, 2, 2, 2, 1, 0, 2])))
        # Check date convertion
        self.assertTrue('sale:year' in ds.dataset.columns)
        self.assertTrue('sale:month' in ds.dataset.columns)
        self.assertTrue('sale:day' in ds.dataset.columns)
        # Check date missing value new column
        self.assertTrue('MachineHoursCurrentMeter_na' in ds.dataset.columns)

    def test_dataset_007(self):
        ''' Test preprocess 'drop_zero_std' feature '''
        # Create dataset
        config = Config(os.path.join('tests', 'unit', 'config', 'test_dataset_007.yaml'), argv=list())
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check that columns having zero std are dropped
        self.assertFalse('datasource' in ds.dataset.columns)
        self.assertFalse('auctioneerID' in ds.dataset.columns)

    def test_dataset_augment_001(self):
        ''' Checking dataset augment: PCA '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_augment_001.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check augmented variables
        df = ds.dataset
        self.assertTrue('pca_x_0' in df.columns)
        self.assertTrue('pca_x_1' in df.columns)
        dfpca = df[['pca_x_0', 'pca_x_1']]
        # Sample 0 PCA
        s0 = dfpca.iloc[0, :].values
        s0_exp = np.array([-0.54432562, -0.20570426])
        self.assertTrue(np.linalg.norm(s0_exp - s0) < 0.05, f"Sample 0 PCA:\n\tExpected:{s0_exp}\n\tValue   :{s0}")
        # Sample 1 PCA
        s1 = dfpca.iloc[1, :].values
        s1_exp = np.array([0.60071371, -0.24420421])
        self.assertTrue(np.linalg.norm(s1_exp - s1) < 0.05, f"Sample 1 PCA:\n\tExpected:{s1_exp}\n\tValue   :{s1}")
        # Check PCA covariance
        pca = ds.dataset_augment.pca_augment.sk_pca_by_name['pca_x']
        cov_expected = np.array([[1.09, 0.65], [0.65, 0.5]])  # Covariance matrix expected
        cov = pca.get_covariance()
        cov_diff = np.linalg.norm(cov - cov_expected)
        self.assertTrue(cov_diff < 0.12, f"Expected covarianve differs (difference norm: {cov_diff}). Covariance:\n{cov}\nExpected:\n{cov_expected}")

    def test_dataset_augment_002(self):
        ''' Checking dataset augment: Operations '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_augment_002.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and augment dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check augmented variables
        df = ds.dataset
        # Check that fields exists or don't exists
        for name in ['sub', 'div']:
            self.assertTrue(f"{name}_expr_x1_x2" in df.columns, f"Missing: {name}_expr_x1_x2")
            self.assertFalse(f"{name}_expr_x2_x1" in df.columns, f"Found {name}_expr_x2_x1")
            self.assertFalse(f"{name}_expr_n1_n2" in df.columns, f"Found {name}_expr_n1_n2")
        for name in ['add', 'mult']:
            self.assertTrue(f"{name}_expr_x2_x1" in df.columns, f"Missing: {name}_expr_x2_x1")
            self.assertFalse(f"{name}_expr_x1_x2" in df.columns, f"Found {name}_expr_x1_x2")
            self.assertFalse(f"{name}_expr_n2_n1" in df.columns, f"Found {name}_expr_n2_n1")
        # Log10
        self.assertTrue(f"log10_ratio_expr_x3_x4" in df.columns, f"Missing: log10_ratio_expr_x3_x4")
        self.assertFalse(f"log10_ratio_expr_x1_x2" in df.columns, f"Found log10_ratio_expr_x1_x2")
        self.assertFalse(f"log10_ratio_expr_x5_x6" in df.columns, f"Found log10_ratio_expr_x5_x6")
        # Log (natural)
        self.assertTrue(f"loge_ratio_expr_x3_x4" in df.columns, f"Missing: loge_ratio_expr_x3_x4")
        self.assertFalse(f"loge_ratio_expr_x1_x2" in df.columns, f"Found loge_ratio_expr_x1_x2")
        self.assertFalse(f"loge_ratio_expr_x5_x6" in df.columns, f"Found loge_ratio_expr_x5_x6")
        # Log + 1
        self.assertTrue(f"logep1_ratio_expr_x3_x4" in df.columns, f"Missing: logep1_ratio_expr_x3_x4")
        self.assertFalse(f"logep1_ratio_expr_x1_x2" in df.columns, f"Found logep1_ratio_expr_x1_x2")
        self.assertTrue(f"logep1_ratio_expr_x5_x6" in df.columns, f"Missing: logep1_ratio_expr_x5_x6")
        # Check results
        x1, x2, x3, x4 = -2.3042662810235153, 0.1202582216313, 0.9588706570406521, 0.880524565094321
        self.assertTrue(is_close(x1 + x2, df['add_expr_x2_x1'][0]))
        self.assertTrue(is_close(x1 - x2, df['sub_expr_x1_x2'][0]))
        self.assertTrue(is_close(x1 / x2, df['div_expr_x1_x2'][0]))
        self.assertTrue(is_close(np.log(x3 / x4) / np.log(10), df['log10_ratio_expr_x3_x4'][0]))
        self.assertTrue(is_close(np.log(x3 / x4), df['loge_ratio_expr_x3_x4'][0]))
        self.assertTrue(is_close(np.log((x3 + 1) / (x4 + 1)), df['logep1_ratio_expr_x3_x4'][0]))

    def test_dataset_augment_003(self):
        ''' Checking dataset augment: Filter results having too many zeros '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_augment_003.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and augment dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check augmented variables
        df = ds.dataset
        # Mult_1
        should_be = set(['mult_1_x2_x1', 'mult_1_x3_x1', 'mult_1_x4_x1', 'mult_1_x3_x2',
                         'mult_2_x2_x1', 'mult_2_x3_x1'])
        for i in range(1, 7):
            for j in range(i + 1, 7):
                for m in [1, 2]:
                    name = f"mult_{m}_x{i}_x{j}"
                    if name in should_be:
                        self.assertTrue(name in df.columns, f"Missing augmented column: {name}")
                    else:
                        self.assertFalse(name in df.columns, f"Found augmented column: {name}, should not be there")

    def test_dataset_augment_004(self):
        ''' Checking dataset augment: Add or Multiply more than 2 fields '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_augment_004.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and augment dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check augmented variables
        df = ds.dataset
        # Add (3 fields) and Mult (4 fields)
        should_be = set(['add_3_x3_x2_x1', 'add_3_x4_x2_x1', 'add_3_x5_x2_x1',
                         'add_3_x6_x2_x1', 'add_3_x4_x3_x1', 'add_3_x5_x3_x1',
                         'add_3_x6_x3_x1', 'add_3_x5_x4_x1', 'add_3_x6_x4_x1',
                         'add_3_x6_x5_x1', 'add_3_x4_x3_x2', 'add_3_x5_x3_x2',
                         'add_3_x6_x3_x2', 'add_3_x5_x4_x2', 'add_3_x6_x4_x2',
                         'add_3_x6_x5_x2', 'add_3_x5_x4_x3', 'add_3_x6_x4_x3',
                         'mult_4_x4_x3_x2_x1', 'mult_4_x5_x3_x2_x1', 'mult_4_x6_x3_x2_x1',
                         'mult_4_x5_x4_x2_x1', 'mult_4_x6_x4_x2_x1', 'mult_4_x6_x5_x2_x1',
                         'mult_4_x5_x4_x3_x1', 'mult_4_x6_x4_x3_x1', 'mult_4_x6_x5_x3_x1',
                         'mult_4_x6_x5_x4_x1', 'mult_4_x5_x4_x3_x2', 'mult_4_x6_x4_x3_x2',
                         'mult_4_x6_x5_x3_x2'])
        for f1 in range(1, 7):
            for f2 in range(f1 + 1, 7):
                for f3 in range(f2 + 1, 7):
                    name = f"add_3_x{f3}_x{f2}_x{f1}"
                    if name in should_be:
                        self.assertTrue(name in df.columns, f"Missing augmented column: {name}")
                    else:
                        self.assertFalse(name in df.columns, f"Found augmented column: {name}, should not be there")
                    for f4 in range(f3 + 1, 7):
                        name = f"mult_4_x{f4}_x{f3}_x{f2}_x{f1}"
                        if name in should_be:
                            self.assertTrue(name in df.columns, f"Missing augmented column: {name}")
                        else:
                            self.assertFalse(name in df.columns, f"Found augmented column: {name}, should not be there")

    def test_dataset_augment_005(self):
        ''' Checking dataset augment: Add or Multiply more than 2 fields '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_augment_005.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and augment dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check augmented variables
        df = ds.dataset
        and_3_len = len([n for n in df.columns if n.startswith('and_3')])
        and_4_len = len([n for n in df.columns if n.startswith('and_4')])
        self.assertEqual(20, and_3_len, f"Expecting 20 'and_3' columns, got {and_3_len}")
        self.assertEqual(15, and_4_len, f"Expecting 15 'and_4' columns, got {and_4_len}")

    def test_dataset_feature_importance_001(self):
        ''' Checking feature importance on dataset (dataframe): Clasification test (logistic regression model) '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_001.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        lrw = LogisticRegressionWilks(ds, ['x6'], 'test_dataset_feature_importance_001')
        ret = lrw()
        self.assertTrue(ret)
        self.assertTrue(lrw.p_values['x1'] < 0.05, f"p-value = {lrw.p_values['x1']}")
        self.assertTrue(lrw.p_values['x2'] < 0.05, f"p-value = {lrw.p_values['x2']}")
        self.assertTrue(lrw.p_values['x3'] < 0.05, f"p-value = {lrw.p_values['x3']}")
        self.assertTrue(lrw.p_values['x4'] > 0.1, f"p-value = {lrw.p_values['x4']}")
        self.assertTrue(lrw.p_values['x5'] > 0.1, f"p-value = {lrw.p_values['x5']}")

    def test_dataset_feature_importance_002(self):
        ''' Checking feature importance on dataset (dataframe)
        Model type: clasification
        Feature importance method:
            - Permutation
            - random forest
            - Single iteration
            - No p-value
            - No cross-validation
        '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_002.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        fi = DataFeatureImportance(config, ds, 'classification', 'unit_test')
        ret = fi()
        self.assertTrue(ret)
        # Make sure we can select x1 and x2 as important varaibles
        fi_vars = list(fi.results.df.index.values)
        self.assertTrue(fi_vars[0] == 'x1', f"Feature importance variables are: {fi_vars}")

    def test_dataset_feature_importance_003(self):
        ''' Checking feature importance on dataset (dataframe): Regression test (linear) '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_003.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        fi = DataFeatureImportance(config, ds, 'regression', 'unit_test')
        ret = fi()
        self.assertTrue(ret)
        # Make sure we can select x1 and x2 as important varaibles
        fi_vars = list(fi.results.df.index.values)
        self.assertTrue(fi_vars[0] == 'x1', f"Feature importance variables are: {fi_vars}")
        self.assertTrue(fi_vars[1] == 'x2', f"Feature importance variables are: {fi_vars}")
        self.assertTrue(fi_vars[2] == 'x3', f"Feature importance variables are: {fi_vars}")

    def test_dataset_feature_importance_004(self):
        ''' Checking feature importance on dataset (dataframe): Reression test (linear regression model) '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_004.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        lrw = PvalueLinear(ds, ['x6'], 'test_dataset_feature_importance_004')
        ret = lrw()
        self.assertTrue(ret)
        self.assertTrue(lrw.p_values['x1'] < 0.05, f"p-value = {lrw.p_values['x1']}")
        self.assertTrue(lrw.p_values['x2'] < 0.05, f"p-value = {lrw.p_values['x2']}")
        self.assertTrue(lrw.p_values['x3'] < 0.05, f"p-value = {lrw.p_values['x3']}")
        self.assertTrue(lrw.p_values['x4'] > 0.1, f"p-value = {lrw.p_values['x4']}")
        self.assertTrue(lrw.p_values['x5'] > 0.1, f"p-value = {lrw.p_values['x5']}")

    def test_dataset_feature_importance_005(self):
        ''' Checking feature importance on dataset (dataframe): Clasification test (multi-class logistic regression) '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_005.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        lrw = MultipleLogisticRegressionWilks(ds, ['x4', 'x5'], 'test_dataset_feature_importance_005')
        ret = lrw()
        self.assertTrue(ret)
        self.assertTrue(lrw.p_values['x1'] < 0.05, f"p-value = {lrw.p_values['x1']}")
        self.assertTrue(lrw.p_values['x2'] < 0.05, f"p-value = {lrw.p_values['x2']}")
        self.assertTrue(lrw.p_values['x3'] < 0.05, f"p-value = {lrw.p_values['x3']}")
        self.assertTrue(np.isnan(lrw.p_values['x4']), f"p-value = {lrw.p_values['x4']}")
        self.assertTrue(np.isnan(lrw.p_values['x5']), f"p-value = {lrw.p_values['x5']}")

    def test_dataset_feature_importance_006(self):
        ''' Checking feature importance on dataset (dataframe): Clasification test (multi-class logistic regression) '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_006.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        lrw = MultipleLogisticRegressionWilks(ds, ['x3'], 'test_dataset_feature_importance_005')
        ret = lrw()
        self.assertTrue(ret)
        # P-values
        self.assertTrue(lrw.p_values_corrected[0] < 1e-160, f"p-value = {lrw.p_values_corrected[0]}")
        self.assertTrue(lrw.p_values_corrected[1] < 1e-50, f"p-value = {lrw.p_values_corrected[1]}")
        self.assertTrue(lrw.p_values_corrected[2] == 1.0, f"p-value = {lrw.p_values_corrected[2]}")  # This one is 'x3' which is part of the null model
        self.assertTrue(lrw.p_values_corrected[3] > 0.1, f"p-value = {lrw.p_values_corrected[3]}")
        # Coefficients
        self.assertTrue(lrw.coefficients['x1'] > 3.5, f"coefficients = {lrw.coefficients['x1']}")
        self.assertTrue(lrw.coefficients['x2'] < -1.3, f"coefficients = {lrw.coefficients['x2']}")
        # Best p-vlues referes to categories...
        self.assertTrue(lrw.best_category[0] == 'large', f"coefficients = {lrw.best_category[0]}")
        self.assertTrue(lrw.best_category[1] == 'med', f"coefficients = {lrw.best_category[1]}")
        # self.assertTrue(lrw.best_category[2] == 'med', f"coefficients = {lrw.best_category[2]}")  # This one is 'x3' which is part of the null model
        self.assertTrue(lrw.best_category[3] == 'med', f"coefficients = {lrw.best_category[3]}")

    def test_dataset_feature_importance_007(self):
        '''
        Checking feature importance on dataset (dataframe)
        Model type: clasification
        Feature importance method:
            - Permutation
            - random forest
            - Multiple-iterations
            - Calculate p-value
            - No cross-validation
        '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_007.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        fi = DataFeatureImportance(config, ds, 'classification', 'unit_test')
        ret = fi()
        self.assertTrue(ret)
        # Make sure we can select x1 and x2 as important varaibles
        fi_vars = list(fi.results.df.index.values)
        self.assertTrue(fi_vars[0] == 'x1', f"Feature importance variables are: {fi_vars}")

    def test_dataset_feature_importance_008(self):
        '''
        Checking feature importance on dataset (dataframe)
        Model type: classification
        Feature importance method:
            - Permutation
            - random forest
            - Multiple-iterations
            - Calculate p-value
            - Use cross-validation
        '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_008.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        ds = DatasetsCv(config, ds)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        fi = DataFeatureImportance(config, ds, 'classification', 'unit_test')
        ret = fi()
        self.assertTrue(ret)
        # Make sure we can select x1 and x2 as important varaibles
        fi_vars = list(fi.results.df.index.values)
        self.assertTrue(fi_vars[0] == 'x1', f"Feature importance variables are: {fi_vars}")

    def test_dataset_feature_importance_009(self):
        '''
        Checking feature importance on dataset (dataframe)
        Model type: classification
        Feature importance method:
            - drop-column
            - random forest
            - Multiple-iterations
            - Calculate p-value
            - Use cross-validation
        '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_feature_importance_009.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Load and preprocess dataset
        ds = DatasetsDf(config)
        ds = DatasetsCv(config, ds)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Do feature importance using logistic regression p-values
        fi = DataFeatureImportance(config, ds, 'classification', 'unit_test')
        ret = fi()
        self.assertTrue(ret)
        # Make sure we can select x1 and x2 as important varaibles
        fi_vars = list(fi.results.df.index.values)
        self.assertTrue(fi_vars[0] == 'x1', f"Feature importance variables are: {fi_vars}")

    def test_dataset_preprocess_001(self):
        ''' Checking dataset preprocess for dataframe: Normalization '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_001.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # x1
        self.assertTrue(np.min(df.x1) >= 0.0)
        self.assertTrue(np.max(df.x1) <= 1.0)
        # x2
        self.assertTrue(np.min(df.x2) >= 0.0)
        self.assertTrue(np.max(df.x2) <= 1.0)
        # x3
        self.assertTrue(np.max(df.x3) <= 1.0)
        self.assertTrue(abs(np.min(df.x3)) <= 1.0)
        # x4
        self.assertTrue(np.min(df.x4) >= 0.0)
        self.assertTrue(np.min(df.x4) < 0.0001)
        self.assertTrue(np.max(df.x4) > 0.9999)
        self.assertTrue(np.max(df.x4) <= 1.0)
        # x5
        self.assertTrue(np.min(df.x5) >= -1.0)
        self.assertTrue(np.min(df.x5) < -0.9999)
        self.assertTrue(np.max(df.x5) > 0.9999)
        self.assertTrue(np.max(df.x5) <= 1.0)
        # x6
        x6_mean = np.mean(df.x6)
        x6_std = np.std(df.x6)
        self.assertTrue(abs(x6_mean - 2) < 0.2)
        self.assertTrue(abs(x6_std - 3) <= 0.1)
        # x7
        x7_mean = np.mean(df.x7)
        x7_std = np.std(df.x7)
        self.assertTrue(abs(x7_mean) < 0.001)
        self.assertTrue(abs(x7_std - 1) <= 0.001)

    def test_dataset_preprocess_002(self):
        ''' Checking dataset preprocess for dataframe: Imputation '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_002.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Check results
        epsilon = 0.0000001
        var = 'x0'
        idx = 11
        expected = 0.017394156072708448
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var = 'x2'
        idx = 7
        expected = -0.04392194621855322
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var = 'x3'
        idx = 12
        expected = 1.0
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var = 'x5'
        idx = 17
        expected = 0.0
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var = 'x6'
        idx = 4
        self.assertTrue(np.isnan(df[var][idx]), f"df.{var}[{idx}] = {df[var][idx]}")
        var = 'x10'
        idx = 12
        expected = 2.0
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")

    def test_dataset_preprocess_003(self):
        ''' Checking dataset preprocess for dataframe: Imputation '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_003.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Check results
        epsilon = 0.0000001
        var, idx, expected = 'x1', 996, 0
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'x2', 49, 1
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'z0', 13, 0
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'z1', 17, 0
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'z2', 12, 0
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'o0', 0, 1
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'o1', 5, 1
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'o2', 28, 1
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")
        var, idx, expected = 'oz2', 1, 1
        self.assertTrue(abs(df[var][idx] - expected) < epsilon, f"df.{var}[{idx}] = {df[var][idx]}")

    def test_dataset_preprocess_004(self):
        ''' Checking dataset preprocess for dataframe: Balance '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_004.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config, model_type='classification')
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Check results
        uniq, counts = np.unique(df.y, return_counts=True)
        perc = counts / counts.sum()
        epsilon = 0.0001
        expected = 1.0 / 3.0
        self.assertTrue((np.abs(perc - expected) < epsilon).all(), f"Unbalanced percentage: {perc}, counts={counts}, cathegories={uniq}")

    def test_dataset_preprocess_005(self):
        ''' Checking dataset preprocess: Remove missing output rows '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_005.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        # Check that rows in 'y' have been removed
        self.assertTrue(df.shape[0] < 990)

    def test_dataset_preprocess_006(self):
        ''' Checking dataset preprocess: Remove missing output rows '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_006.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        for c in ['d1:year', 'd1:month', 'd1:week', 'd1:day', 'd1:dayofweek', 'd1:dayofyear', 'd1:is_month_end', 'd1:is_month_start', 'd1:is_quarter_end', 'd1:is_quarter_start', 'd1:is_year_end', 'd1:is_year_start', 'd1:hour', 'd1:minute', 'd1:second', 'd1:elapsed']:
            x = df[c]
            self.assertTrue(x.isna().sum() == 0, f"Column {c} has {x.isna().sum()} missing elements")

    def test_dataset_preprocess_007(self):
        ''' Checking dataset preprocess: Remove missing column '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_007.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        cols = list(ds.dataset.columns)
        self.assertTrue('x2' in cols)
        self.assertTrue('y' in cols)
        self.assertFalse('x1' in cols)
        self.assertFalse('d1' in cols)

    def test_dataset_preprocess_008(self):
        ''' Checking dataset preprocess: Convert fields to categories (matching regex on field names) '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_008.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        cols = list(ds.dataset.columns)
        self.assertTrue('x1' in cols)
        self.assertTrue('x2' in cols)
        for field in ['zzz_0', 'zzz_2', 'zzz_4', 'zzz_6', 'zzz_8', 'zxz_1:high', 'xzz_3:high', 'azzz_5:high', '_zzz_7:high', 'zzzz_9']:
            self.assertTrue(field in cols, f"Field {field} not found")

    def test_dataset_preprocess_009(self):
        ''' Checking dataset preprocess: Remove duplicate inputs '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_009.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        cols = list(ds.dataset.columns)
        self.assertTrue('x1' in cols)
        self.assertTrue('x2' in cols)
        self.assertTrue('x3' in cols)
        self.assertFalse('x1r' in cols)
        self.assertFalse('x2r' in cols)
        self.assertFalse('x3r' in cols)

    def test_dataset_preprocess_010(self):
        ''' Checking dataset preprocess: Shuffle data '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_010.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        cols = list(df.columns)
        self.assertFalse(is_sorted(df.x1.to_numpy()), f"Data should not be sorted:\ndf.x1={df.x1}")
        self.assertFalse(is_sorted(df.x2.to_numpy()), f"Data should not be sorted:\ndf.x2={df.x2}")
        self.assertFalse(is_sorted(df.y.to_numpy()), f"Data should not be sorted:\ndf.y={df.y}")

    def test_dataset_preprocess_011(self):
        ''' Checking dataset preprocess: Binary categorical data with NAs as -1 '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_011.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        cols = list(df.columns)
        col, c_min, c_max, unique_expected = 'x1', -1, 0, [-1, 0]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'z2', 0, 2, [0, 1, 2]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'x3', -1, 1, [-1, 0, 1]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'a4', 0, 3, [0, 1, 2, 3]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'y', 0, 2, [0, 1, 2]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")

    def test_dataset_preprocess_012(self):
        ''' Checking dataset preprocess: Binary categorical data with NAs as -1 '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_012.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        cols = list(df.columns)
        col, c_min, c_max, unique_expected = 'x1', 0, 0, [0, np.nan]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'z2', 0.5, 1, [0.5, 1, np.nan]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'x3', 0, 1, [0, 1, np.nan]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'a4', 1 / 3, 1, [1 / 3, 2 / 3, 3 / 3, np.nan]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'y', 0, 2, [0, 1, 2]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")

    def test_dataset_preprocess_013(self):
        ''' Checking dataset preprocess: Binary categorical data with NAs as -1 '''
        config_file = os.path.join('tests', 'unit', 'config', 'test_dataset_preprocess_013.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        df = ds.dataset
        cols = list(df.columns)
        col, c_min, c_max, unique_expected = 'x1', -1, 0, [-1, 0]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'z2', 0, 1, [0, 0.5, 1]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'x3', -1, 1, [-1, 0, 1]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'a4', 0, 1, [0, 1 / 3, 2 / 3, 1]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")
        col, c_min, c_max, unique_expected = 'y', 0, 2, [0, 1, 2]
        self.assertTrue(df[col].min() == c_min, f"Minimum {col} is not {c_min}, it's {df[col].min()}: {df[col].values}")
        self.assertTrue(df[col].max() == c_max, f"Maximum {col} is not {c_max}, it's {df[col].max()}: {df[col].values}")
        uniq = np.sort(df[col].unique())
        self.assertTrue(array_equal(unique_expected, uniq), f"Unique values '{col}' expected {unique_expected}, but got {uniq}")

    def test_files_001(self):
        mf = MlFiles()
        data = {'a': random.random(), 'b': random.random(), 'c': random.random()}
        file_pkl = os.path.join('tests', 'tmp', 'test_files_001.pkl')
        mf._save_pickle(file_pkl, 'Test', data)
        data_load = mf._load_pickle(file_pkl, 'Test')
        self.assertEqual(data, data_load)

    def test_files_002(self):
        mf = MlFiles()
        data_load = mf._load_pickle('', 'Test')
        self.assertEqual(data_load, None)

    def test_files_003(self):
        mf = MlFiles()
        file_pkl = os.path.join('tests', 'tmp', f"test_files_{random.random()}.pkl")
        data_load = mf._load_pickle(file_pkl, 'Test')
        self.assertEqual(data_load, None)

    def test_files_004(self):
        mf = MlFiles()
        ret = mf._save_pickle('', 'Test', [1, 2, 3])
        self.assertEqual(ret, False)

    def test_files_005(self):
        mf = MlFiles()
        data = {'a': random.random(), 'b': random.random(), 'c': random.random()}
        file_yaml = os.path.join('tests', 'tmp', 'test_files_005.yaml')
        mf._save_yaml(file_yaml, data)
        data_load = mf._load_yaml(file_yaml)
        self.assertEqual(data, data_load)

    def test_log_001(self):
        mllog = MlLog()
        # Files for tee
        mllog.file_stdout = os.path.join('tests', 'tmp', f"test_log_001.stdout")
        mllog.file_stderr = os.path.join('tests', 'tmp', f"test_log_001.stderr")
        # Delete old files
        rm(mllog.file_stdout)
        rm(mllog.file_stderr)
        # Open tee, print messages, close tee
        mllog.tee()
        test_out = "TEST test_log_001: STDOUT"
        test_err = "TEST test_log_001: STDERR"
        print(test_out)
        print(test_err, file=sys.stderr)
        mllog.tee(close=True)
        # Check files
        with open(mllog.file_stdout) as f:
            self.assertEqual(f.read(), test_out + '\n')
        with open(mllog.file_stderr) as f:
            self.assertEqual(f.read(), test_err + '\n')

    def test_train_001(self):
        ''' Check Model.__call__() '''
        def test_train_001_dataset_create(num_create):
            assert num_create == 42
            ds = np.arange(100)
            return ds

        def test_train_001_dataset_inout(d):
            return d, d

        def test_train_001_model_create(x, y, beta):
            assert beta == 0.1
            return {'mean': 0}

        def test_train_001_model_train(model, x, y, epochs, lr):
            assert lr == 0.1
            assert epochs == 100
            mean = np.array(x).mean()
            model['mean'] = mean
            return mean

        def test_train_001_model_evaluate(model, x, y, param):
            assert param == 42
            return model['mean']

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_train_001_dataset_create)
        MlRegistry().register(DATASET_INOUT, test_train_001_dataset_inout)
        MlRegistry().register(MODEL_CREATE, test_train_001_model_create)
        MlRegistry().register(MODEL_TRAIN, test_train_001_model_train)
        MlRegistry().register(MODEL_EVALUATE, test_train_001_model_evaluate)
        # Read config
        config_file = os.path.join('tests', 'unit', 'config', 'test_train_001.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Cleanup old files
        rm(os.path.join('tests', 'tmp', 'test_train_001.pkl'))
        # Create LogMl
        random.seed(20190705)
        logml = LogMl(config=config)
        ret = logml()
        # Check values
        mltrain = logml.model
        model_expected = {'mean': 44.88095238095238}
        model_pkl = os.path.join('tests', 'tmp', 'model', f"model_test_train_001.model.{mltrain._id}.pkl")
        test_results_pkl = os.path.join('tests', 'tmp', 'model', f"model_test_train_001.test_results.{mltrain._id}.pkl")
        train_results_pkl = os.path.join('tests', 'tmp', 'model', f"model_test_train_001.train_results.{mltrain._id}.pkl")
        self.assertTrue(ret)
        self.assertEqual(mltrain.model, model_expected)
        self.assertEqual(logml._load_pickle(model_pkl, 'model_load'), model_expected)
        self.assertEqual(mltrain.load_test_results(), model_expected['mean'])
        self.assertEqual(logml._load_pickle(train_results_pkl, 'train_results'), model_expected['mean'])

    def test_train_002_cross_validate(self):
        ''' Check Model.__call__() '''
        def test_train_002_dataset_create(num_create):
            assert num_create == 42
            ds = np.arange(100)
            return ds

        def test_train_002_dataset_inout(d):
            return d, d

        def test_train_002_model_create(x, y, beta):
            assert beta == 0.1
            return {'mean': 0}

        def test_train_002_model_train(model, x, y, epochs, lr):
            assert lr == 0.1
            assert epochs == 100
            mean = np.array(x).mean()
            model['mean'] = mean
            return mean

        def test_train_002_model_evaluate(model, x, y, param):
            assert param == 42
            m = model['mean']
            return np.sqrt(np.mean((x - m)**2))

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_train_002_dataset_create)
        MlRegistry().register(DATASET_INOUT, test_train_002_dataset_inout)
        MlRegistry().register(MODEL_CREATE, test_train_002_model_create)
        MlRegistry().register(MODEL_TRAIN, test_train_002_model_train)
        MlRegistry().register(MODEL_EVALUATE, test_train_002_model_evaluate)
        # Read config
        config_file = os.path.join('tests', 'unit', 'config', 'test_train_002_cross_validate.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Cleanup old files
        rm(os.path.join('tests', 'tmp', 'test_train_002_cross_validate.pkl'))
        # Create LogMl
        random.seed(20190706)
        logml = LogMl(config=config)
        ret = logml()
        # Check values
        cv_model = logml.model
        self.assertTrue(ret)
        self.assertEqual(cv_model.eval_validate, 37.134368623752614)
        self.assertEqual(cv_model.eval_validate_std, 19.504326451560278)

    def test_train_003_hyper_opt(self):
        ''' Check hyper-parameter optimization '''
        def test_train_003_dataset_create(num_create):
            assert num_create == 42
            ds = np.arange(10)
            return ds

        def test_train_003_dataset_inout(d):
            return d, d

        def test_train_003_model_create(x, y, beta):
            assert beta == 0.1
            return {'mean': 0}

        def test_train_003_model_train(model, x, y, mean):
            ''' Model train: No training, it only sets the 'mean' from hyper parameter '''
            model['mean'] = mean
            x = np.array(x)
            rmse = np.sqrt(np.mean((x - mean)**2))
            return rmse

        def test_train_003_model_evaluate(model, x, y, param):
            assert param == 42
            mean = model['mean']
            x = np.array(x)
            rmse = np.sqrt(np.mean((x - mean)**2))
            return rmse

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_train_003_dataset_create)
        MlRegistry().register(DATASET_INOUT, test_train_003_dataset_inout)
        MlRegistry().register(MODEL_CREATE, test_train_003_model_create)
        MlRegistry().register(MODEL_TRAIN, test_train_003_model_train)
        MlRegistry().register(MODEL_EVALUATE, test_train_003_model_evaluate)
        # Read config
        config_file = os.path.join('tests', 'unit', 'config', 'test_train_003_hyper_opt.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Cleanup old files
        rm(os.path.join('tests', 'tmp', 'test_train_003_hyper_opt.pkl'))
        # Create LogMl
        logml = LogMl(config=config)
        ret = logml()
        # Check values
        ho = logml.hyper_parameter_optimization
        mean = ho.best['mean']
        self.assertTrue(ret)
        # Check that the result is arround 5
        self.assertTrue(abs(mean - 4.5) < 0.5)

    def test_train_004_hyper_opt_create(self):
        ''' Check hyper-parameter optimization for 'dataset create' parameters '''
        def test_train_004_dataset_create(num_create):
            ds = np.arange(num_create)
            return ds

        def test_train_004_dataset_inout(d):
            return d, d

        def test_train_004_model_create(x, y, beta):
            assert beta == 0.1
            return {'len': 0}

        def test_train_004_model_train(model, x, y, mean):
            ''' Model train: No training, it only sets the dataset length from hyper parameter '''
            return 1.0 - len(x) / 10.0

        def test_train_004_model_evaluate(model, x, y, param):
            return 1.0 - len(x) / 10.0

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_train_004_dataset_create)
        MlRegistry().register(DATASET_INOUT, test_train_004_dataset_inout)
        MlRegistry().register(MODEL_CREATE, test_train_004_model_create)
        MlRegistry().register(MODEL_TRAIN, test_train_004_model_train)
        MlRegistry().register(MODEL_EVALUATE, test_train_004_model_evaluate)
        # Read config
        config_file = os.path.join('tests', 'unit', 'config', 'test_train_004_hyper_opt_create.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Cleanup old files
        rm(os.path.join('tests', 'tmp', 'test_train_004_hyper_opt.pkl'))
        # Create LogMl
        logml = LogMl(config=config)
        ret = logml()
        # Check values
        ho = logml.hyper_parameter_optimization
        dlen = ho.best['num_create']
        self.assertTrue(ret)
        # Check that the result is arround 5
        self.assertEqual(dlen, 87)

    def test_train_005_metrics(self):
        ''' Check Model.__call__() with a custom metric '''
        def test_train_005_dataset_create(num_create):
            assert num_create == 42
            ds = np.arange(9) + 1
            return ds

        def test_train_005_dataset_inout(d):
            return d, d

        def test_train_005_model_create(x, y, beta):
            assert beta == 0.1
            return {'mean': 0}

        def test_train_005_model_predict(model, x):
            return np.arange(9)

        def test_train_005_model_train(model, x, y, epochs, lr):
            assert lr == 0.1
            assert epochs == 100
            mean = np.array(x).mean()
            model['mean'] = mean
            return mean

        # Register functions
        MlRegistry().register(DATASET_CREATE, test_train_005_dataset_create)
        MlRegistry().register(DATASET_INOUT, test_train_005_dataset_inout)
        MlRegistry().register(MODEL_CREATE, test_train_005_model_create)
        MlRegistry().register(MODEL_PREDICT, test_train_005_model_predict)
        MlRegistry().register(MODEL_TRAIN, test_train_005_model_train)
        # Read config
        config_file = os.path.join('tests', 'unit', 'config', 'test_train_005_metrics.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Cleanup old files
        rm(os.path.join('tests', 'tmp', 'test_train_005_metrics.pkl'))
        # Create LogMl
        random.seed(20190705)
        logml = LogMl(config=config)
        ret = logml()
        # Check values
        mltrain = logml.model
        eval_expected = 0.15
        epsilon = 0.000001
        self.assertTrue(ret)
        self.assertTrue(abs(mltrain.eval_validate - eval_expected) < epsilon)


if __name__ == '__main__':
    unittest.main()
