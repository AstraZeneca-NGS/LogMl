
import logging
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
from logml.logml import LogMl
from logml.models import Model
from logml.core.registry import MlRegistry, DATASET_AUGMENT, DATASET_CREATE, DATASET_INOUT, DATASET_PREPROCESS, DATASET_SPLIT, MODEL_CREATE, MODEL_EVALUATE, MODEL_TRAIN


class TestLogMlIntegration(unittest.TestCase):

    def setUp(self):
        MlLog().set_log_level(logging.CRITICAL)
        # MlLog().set_log_level(logging.DEBUG)
        MlRegistry().reset()

    def test_linear3(self):
        config_file = os.path.join('tests', 'integration', 'config' , 'linear3.yaml')
        config = Config(argv=['logml.py', '-c', config_file])
        ret = config()
        ml = LogMl(config=config)
        ml()
        pass

    # def test_linear3c(self):
    #     pass
    #
    # def test_class3(self):
    #     pass
    #
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
