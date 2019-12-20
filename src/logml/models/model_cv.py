
import numpy as np

from ..core.config import CONFIG_CROSS_VALIDATION, CONFIG_MODEL
from ..core.log import MlLog
from . import Model
from ..datasets import DatasetsCv


def _id(obj):
    return id(obj) if obj is not None else 'None'


class ModelCv(Model):
    '''
    A Model with cross-validation capabilities
    '''
    def __init__(self, config, datasets=None, set_config=True):
        super().__init__(config, datasets, set_config)
        # Get cross-validation parameters
        self.cv_config = self.config.get_parameters(CONFIG_CROSS_VALIDATION)
        self.cv_enable = self.cv_config.get('enable', False)
        self.cv_models = list()   # A list of models for each cross-validation
        self.eval_train_std = None
        self.eval_validate_std = None
        self.cv_count = 0
        self.datasets_cv = datasets
        if self.cv_enable:
            self.cv_count = self.datasets_cv.cv_count
            self.datasets_cv = datasets

    def _cross_validate_f(self, f, collect_name=None, args=None):
        """
        Run cross-validation evaluating function 'f' and collecting field 'collect_name'
        Returns a tuple of two lists: (rets, collects)
            - rest: All return values from each f() invokation
            - collects: All collected values, after each f() invokation
        """
        # Replace datasets for each cross-validation datasets
        datasets_ori, model_ori = self.datasets, self.model
        # Initialize
        rets = list()
        collect = list()
        for d, m in zip(self.datasets_cv, self):
            # Evaluate model (without cross-validation) on cv_dataset[i]
            self.datasets, self.model = d, m
            self._debug(f"Cross-validation: Invoking function '{f.__name__}', dataset.id={_id(self.datasets)}, model.id={_id(self.model)}")
            if args is None:
                rets.append(f())
            else:
                rets.append(f(*args))
            if collect_name is not None:
                collect.append(self.__dict__[collect_name])
        # Restore original datasets and model
        self.datasets, self.model = datasets_ori, model_ori
        return rets, collect

    def __iter__(self):
        """ Iterate over all datasets in cross-validation """
        return self.cv_models.__iter__()

    def model_create(self):
        """ Create model for cross-validation """
        if not self.cv_enable:
            return super().model_create()
        self.cv_models = [None] * self.cv_count
        rets, self.cv_models = self._cross_validate_f(super().model_create, 'model')
        return all(rets)

    def model_eval_test(self):
        """ Evaluate model on 'test' dataset, using cross-validation """
        if not self.cv_enable:
            return super().model_eval_test()
        rets, losses = self._cross_validate_f(super().model_eval_test, 'eval_test')
        losses = np.array(losses)
        self.eval_test, self.eval_test_std = losses.mean(), losses.std()
        losses_str = np.array2string(losses, max_line_width=np.inf)
        self._debug(f"Model eval test (cross-validation): losses={losses_str}, eval_test={self.eval_test}, eval_test_std={self.eval_test_std}")
        return all(rets)

    def model_eval_train(self):
        """ Evaluate model on 'train' dataset, using cross-validation """
        if not self.cv_enable:
            return super().model_eval_train()
        rets, losses = self._cross_validate_f(super().model_eval_train, 'eval_train')
        losses = np.array(losses)
        self.eval_train, self.eval_train_std = losses.mean(), losses.std()
        losses_str = np.array2string(losses, max_line_width=np.inf)
        self._debug(f"Model eval train (cross-validation): losses={losses_str}, eval_train={self.eval_train}, eval_train_std={self.eval_train_std}")
        return all(rets)

    def model_eval_validate(self):
        """ Evaluate model on 'validate' dataset, using cross-validation """
        if not self.cv_enable:
            return super().model_eval_validate()
        rets, losses = self._cross_validate_f(super().model_eval_validate, 'eval_validate')
        losses = np.array(losses)
        self.eval_validate, self.eval_validate_std = losses.mean(), losses.std()
        losses_str = np.array2string(losses, max_line_width=np.inf)
        self._debug(f"Model eval validate (cross-validation): losses={losses_str}, eval_validate={self.eval_validate}, eval_validate_std={self.eval_validate_std}")
        return all(rets)

    def model_train(self):
        """ Train models for cross-validation """
        if not self.cv_enable:
            return super().model_train()
        rets, _ = self._cross_validate_f(super().model_train, None)
        return all(rets)
