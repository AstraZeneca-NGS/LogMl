
import logging
import numpy as np
import pandas as pd
import os
import random
import sys
import unittest

from logml.core.config import Config, CONFIG_CROSS_VALIDATION, CONFIG_DATASET, CONFIG_HYPER_PARAMETER_OPTMIMIZATION, CONFIG_LOGGER, CONFIG_MODEL
from logml.datasets import Datasets, DatasetsDf
from logml.core.files import MlFiles
from logml.core.log import MlLog
from logml.logml import LogMl
from logml.models import Model
from logml.core.registry import MlRegistry, DATASET_AUGMENT, DATASET_CREATE, DATASET_INOUT, DATASET_PREPROCESS, DATASET_SPLIT, MODEL_CREATE, MODEL_EVALUATE, MODEL_TRAIN


# Create dataset
def create_dataset_preprocess_001():
    # Number of samples
    num = 1000
    # Inputs: x1, .. ., xn
    x1 = np.exp(np.random.rand(num))
    x2 = np.maximum(np.random.rand(num) - 0.1, 0)
    x3 = np.random.normal(0, 1, num)
    x4 = np.random.rand(num) * 5 + 7
    x5 = np.random.rand(num) * 5 + 7
    x6 = np.random.normal(2, 3, num)
    x7 = np.random.normal(3, 4, num)
    x8 = np.random.rand(num) * 2 + 3
    # Noise
    n = np.random.normal(0, 1, num)
    # Output
    y = 3. * x1 - 1. * x2 + 0.5 * x3 + 0.1 * n
    # Categorical output
    y_str = np.array([to_class(c) for c in y])
    # Create dataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6, 'x7': x7, 'y': y_str})
    file = 'test_dataset_preprpocess_001.csv'
    print(f"Saving dataset to file '{file}'")
    df.to_csv(file, index=False)
    return df


def rm(file):
    ''' Delete file, if it exists '''
    if os.path.exists(file):
        os.remove(file)


def to_class(c):
    if c < -3:
        return 'low'
    if c < 3:
        return 'mid'
    return 'high'


class TestLogMl(unittest.TestCase):

    def setUp(self):
        MlLog().set_log_level(logging.CRITICAL)
        # MlLog().set_log_level(logging.DEBUG)
        MlRegistry().reset()

    def test_config_001(self):
        ''' Test objects parameters from config and file names '''
        config_file = os.path.join('tests', 'ml.test_config_001.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        ret = config()
        mldataset = Datasets(config)
        logml = LogMl(config=config)
        mltrain = Model(config, mldataset)
        self.assertTrue(ret)
        self.assertEqual(mltrain.model_name, 'model_001')
        self.assertEqual(mldataset.get_file_name(), os.path.join('tests', 'tmp', 'test_001.pkl'))
        self.assertEqual(mltrain.get_file_name(), os.path.join('tests', 'tmp', 'model', f"model_001.{mltrain._id}.pkl"))
        self.assertEqual(logml.cross_validation.enable, False)
        self.assertEqual(logml.hyper_parameter_optimization.enable, False)
        self.assertEqual(config.get_parameters_functions('dataset_augment'), {'num_augment': 10})

    def test_config_002(self):
        ''' Test sanity check: hyper-param and cross-validation both enabled '''
        config = Config(os.path.join('tests', 'ml.test_config_002.yaml'), argv=list())
        ret = config()
        config.exit_on_fatal_error = False
        logml = LogMl(config=config)
        self.assertEqual(logml._config_sanity_check(), False)
        self.assertEqual(logml.cross_validation.enable, True)
        self.assertEqual(logml.hyper_parameter_optimization.enable, True)

    def test_config_003(self):
        config = Config(os.path.join('tests', 'ml.test_config_003.yaml'), argv=list())
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
        config = Config(os.path.join('tests', 'ml.test_dataset_001.yaml'), argv=list())
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
        config = Config(os.path.join('tests', 'ml.test_dataset_002.yaml'), argv=list())
        config()
        # Create dataset
        ds = Datasets(config)
        # Cleanup old files
        rm(ds.get_file_name())
        ret = ds()
        # Check values
        self.assertTrue(ret)
        self.assertEqual(ds.operations_done, set([DATASET_AUGMENT, DATASET_PREPROCESS, DATASET_SPLIT]))
        self.assertEqual(ds.dataset, [8, 9, 10, 11, 12, 13])
        self.assertEqual(ds.get_train(), [1, 2, 3])
        self.assertEqual(ds.get_validate(), [4, 5])
        self.assertEqual(ds.get_test(), [6])
        # Load from pickle and check values
        ds2 = Datasets(config)
        ret = ds2.load()
        self.assertTrue(ret)
        self.assertEqual(ds.dataset, [8, 9, 10, 11, 12, 13])
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
        config = Config(os.path.join('tests', 'ml.test_dataset_003.yaml'), argv=list())
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
        config = Config(os.path.join('tests', 'ml.test_dataset_004.yaml'), argv=list())
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
        config = Config(os.path.join('tests', 'ml.test_dataset_005.yaml'), argv=list())
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
        config = Config(os.path.join('tests', 'ml.test_dataset_006.yaml'), argv=list())
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check categories
        self.assertTrue(all(ds.dataset['UsageBand'].head(10) == np.array([1, 1, 1, 0, 1, 1, 1, 0, -1, 1])))
        # Check date convertion
        self.assertTrue('saleYear' in ds.dataset.columns)
        self.assertTrue('saleMonth' in ds.dataset.columns)
        self.assertTrue('saleDay' in ds.dataset.columns)
        # Check date missing value new column
        self.assertTrue('MachineHoursCurrentMeter_na' in ds.dataset.columns)

    def test_dataset_007(self):
        ''' Test DfTransform 'drop_zero_std' feature '''
        # Create dataset
        config = Config(os.path.join('tests', 'ml.test_dataset_007.yaml'), argv=list())
        config()
        ds = DatasetsDf(config)
        rm(ds.get_file_name())
        ret = ds()
        self.assertTrue(ret)
        # Check that columns having zero std are dropped
        self.assertFalse('datasource' in ds.dataset.columns)
        self.assertFalse('auctioneerID' in ds.dataset.columns)

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
        config_file = os.path.join('tests', 'ml.test_train_001.yaml')
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
        config_file = os.path.join('tests', 'ml.test_train_002_cross_validate.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        config()
        # Cleanup old files
        rm(os.path.join('tests', 'tmp', 'test_train_002_cross_validate.pkl'))
        # Create LogMl
        random.seed(20190706)
        logml = LogMl(config=config)
        ret = logml()
        # Check values
        cv = logml.cross_validation
        self.assertTrue(ret)
        self.assertEqual(cv.mean, 37.134368623752614)
        self.assertEqual(cv.std, 19.504326451560278)

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
        config_file = os.path.join('tests', 'ml.test_train_003_hyper_opt.yaml')
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
        config_file = os.path.join('tests', 'ml.test_train_004_hyper_opt_create.yaml')
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

    def test_dataset_preprocess_001(self):
        ''' Checking dataset preprocess for dataframe '''
        config_file = os.path.join('tests', 'ml.test_dataset_preprocess_001.yaml')
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


if __name__ == '__main__':
    unittest.main()
