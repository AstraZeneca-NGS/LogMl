import inspect
import sys
import traceback

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import RidgeCV

from .model_cv import ModelCv
from ..core import MODEL_TYPE_CLASSIFICATION
from ..core import MODEL_TYPE_REGRESSION
from ..util.etc import camel_to_snake


class SkLearnModel(ModelCv):
    """ Create a wrapper for a SkLearn model """
    def __init__(self, config, datasets, class_name, params, set_config=True):
        super().__init__(config, datasets, set_config)
        self.is_save_params = False
        # Set model specific paramters
        self.class_name = class_name
        model_name = camel_to_snake(class_name)
        self._debug(f"Class name: '{class_name}', model name: '{model_name}'")
        if set_config:
            self._set_from_config()
        # Set parameters
        if params:
            for n in params:
                self.__dict__[n] = params[n]
                self._debug(f"Setting {n} = {params[n]}")

    def clone(self):
        """ Clone the model """
        model_clone = super().clone()
        if model_clone.model is not None:
            model_clone.model = clone(model_clone.model)
        return model_clone

    def default_model_create(self, x, y):
        """ Create real model from SciKit learn """
        self._info(f"Creating model based on class '{self.class_name}'")
        class_reference = eval(self.class_name)
        args_spec = inspect.getargspec(class_reference.__init__)
        args_init = args_spec.args
        self._debug(f"Class '{self.class_name}' has constructor arguments: {args_init}")
        kwargs = dict()
        for arg in args_init:
            if arg in self.__dict__:
                val = self.__dict__.get(arg)
                kwargs[arg] = val
        self._debug(f"Invoking constructor '{self.class_name}', with arguments: {kwargs}")
        self.model = eval(f"{self.class_name}(**kwargs)")
        return True

    def default_model_predict(self, x):
        """ Default implementation for '@model_predict' """
        try:
            self._debug(f"Model predict ({self.class_name}): Start, x.shape={x.shape}")
            y_hat = self.model.predict(x)
            self._debug(f"Model predict ({self.class_name}): End")
            return y_hat
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            return None

    def default_model_train(self, x, y):
        """ Fit the model using training data """
        try:
            self._debug(f"Model train ({self.class_name}): Start, x.shape={x.shape}, y.shape={y.shape}")
            self.train_results = self.model.fit(x, y)
            self._debug(f"Model train ({self.class_name}): End")
            return True
        except Exception as e:
            self._error(f"Exception: {e}\n{traceback.format_exc()}")
            traceback.print_stack()
            return False

    def get_feature_importances(self):
        if not self.cv_enable:
            return self.model.feature_importances_
        else:
            # Calculate the average of all improtances
            rets = [m.feature_importances_ for m in self]
            return sum(rets) / len(rets)

    def invoke_model_create(self, x, y):
        return False    # We don't want to call user functions

    def invoke_model_evaluate(self, x, y, name):
        return (False, None)    # We don't want to call user functions

    def invoke_model_predict(self, x):
        return (False, None)    # We don't want to call user functions

    def invoke_model_save(self):
        return False    # We don't want to call user functions

    def invoke_model_train(self, x, y):
        return False    # We don't want to call user functions

    def loss_(self, x, y):
        """ Return the loss """
        ret = super().loss_(x, y)
        if ret is not None:
            return ret
        # Use sklearn model's 'score'
        score = self.model.score(x, y)
        loss = 1.0 - score
        self._debug(f"Loss: Using sklearn score function, score={score}, loss={loss}")
        return loss


class ModelSkExtraTreesRegressor(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100):
        super().__init__(config, datasets, class_name='ModelSkExtraTreesRegressor', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators

    def default_model_create(self, x, y):
        self.model = ExtraTreesRegressor(n_jobs=self.n_jobs, n_estimators=self.n_estimators)
        return True


class ModelSkExtraTreesClassifier(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100):
        super().__init__(config, datasets, class_name='ModelSkExtraTreesClassifier', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators

    def default_model_create(self, x, y):
        self.model = ExtraTreesClassifier(n_jobs=self.n_jobs, n_estimators=self.n_estimators)
        return True


class ModelSkGradientBoostingRegressor(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkGradientBoostingRegressor', params=None, set_config=False)

    def default_model_create(self, x, y):
        self.model = GradientBoostingRegressor()
        return True


class ModelSkGradientBoostingClassifier(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkGradientBoostingClassifier', params=None, set_config=False)

    def default_model_create(self, x, y):
        self.model = GradientBoostingClassifier()
        return True


class ModelSkLarsCV(SkLearnModel):
    def __init__(self, config, datasets, cv):
        super().__init__(config, datasets, class_name='ModelSkLarsCV', params=None, set_config=False)
        self.cv = cv

    def default_model_create(self, x, y):
        self.model = LarsCV(cv=self.cv)
        return True


class ModelSkLassoCV(SkLearnModel):
    def __init__(self, config, datasets, cv):
        super().__init__(config, datasets, class_name='ModelSkLassoCV', params=None, set_config=False)
        self.cv = cv

    def default_model_create(self, x, y):
        self.model = LassoCV(cv=self.cv)
        return True


class ModelSkLassoLarsAIC(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkLassoLarsAIC', params=None, set_config=False)

    def default_model_create(self, x, y):
        self.model = LassoLarsIC(criterion='aic')
        return True


class ModelSkLassoLarsBIC(SkLearnModel):
    def __init__(self, config, datasets):
        super().__init__(config, datasets, class_name='ModelSkLassoLarsBIC', params=None, set_config=False)

    def default_model_create(self, x, y):
        self.model = LassoLarsIC(criterion='bic')
        return True


class ModelSkLassoLarsCV(SkLearnModel):
    def __init__(self, config, datasets, cv):
        super().__init__(config, datasets, class_name='ModelSkLassoLarsCV', params=None, set_config=False)
        self.cv = cv

    def default_model_create(self, x, y):
        self.model = LassoLarsCV(cv=self.cv)
        return True


class ModelSkRandomForestRegressor(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100, max_depth=None, bootstrap=True):
        super().__init__(config, datasets, class_name='ModelSkRandomForestRegressor', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def default_model_create(self, x, y):
        self.model = RandomForestRegressor(n_jobs=self.n_jobs, n_estimators=self.n_estimators, max_depth=self.max_depth, bootstrap=self.bootstrap)
        return True


class ModelSkRandomForestClassifier(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100, max_depth=None, class_weight='balanced', bootstrap=True):
        super().__init__(config, datasets, class_name='ModelSkRandomForestClassifier', params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.bootstrap = bootstrap

    def default_model_create(self, x, y):
        self.model = RandomForestClassifier(n_jobs=self.n_jobs, n_estimators=self.n_estimators, max_depth=self.max_depth, class_weight=self.class_weight, bootstrap=self.bootstrap)
        return True


class ModelSkRidgeCV(SkLearnModel):
    def __init__(self, config, datasets, cv):
        super().__init__(config, datasets, class_name='ModelSkRidgeCV', params=None, set_config=False)
        self.cv = cv

    def default_model_create(self, x, y):
        self.model = RidgeCV(cv=self.cv)
        return True


# TODO: change the name of class
class ModelFactory:
    """ A simple 'factory' class """
    def __init__(self, config, datasets, model_type, model_name, cv_enable, n_estimators=100, **kwargs):
        self.__dict__.update(kwargs)
        self.config = config
        self.datasets = datasets
        self.model_type = model_type
        self.model_name = model_name
        self.cv_enable = cv_enable
        self.n_estimators = n_estimators
        self.model = None

    def get(self, force=False):
        if self.model is None or force:
            self.model = self._fit()
        return self.model

    def is_classification(self):
        return self.model_type == MODEL_TYPE_CLASSIFICATION

    def is_regression(self):
        return self.model_type == MODEL_TYPE_REGRESSION

    def _fit(self):
        """ Create a ExtraTrees model """
        m = ExpandSkLearnModel(self.config, self.datasets, n_jobs=-1, n_estimators=self.n_estimators, model_name=self.model_name)
        if self.cv_enable is not None:
            m.cv_enable = self.cv_enable
        m.model_create()
        m.model_train()

        if m.model is None:
            model_init_params = m._prepare_sklearn_model_class_params()
            # initiate sklearn model with all parameters provided from config
            m.model = m.str_to_class(m.model_name)(**model_init_params)

        return m


class ExpandSkLearnModel(SkLearnModel):
    def __init__(self, config, datasets, n_jobs=-1, n_estimators=100, model_name='', max_depth=None, class_weight='balanced', bootstrap=True):
        super().__init__(config, datasets, class_name=model_name, params=None, set_config=False)
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.model_name = model_name
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.model = None

    def default_model_create(self, x, y):
        model_init_params = self._prepare_sklearn_model_class_params()
        # initiate sklearn model with all parameters provided from config
        self.model = self.str_to_class(self.model_name)(**model_init_params)
        return True

    def _prepare_sklearn_model_class_params(self):
        import inspect
        # get list of all possible sklearn class parameters
        signature = inspect.signature(self.str_to_class(self.model_name))
        sklearn_model_init_params = signature.parameters.keys()

        # check if parameters from config is present in current sklearn class
        model_init_params = dict()
        for parameter in self.__dict__.keys():
            if parameter in sklearn_model_init_params:
                model_init_params[parameter] = self.__dict__[parameter]

        return model_init_params

    @staticmethod
    def str_to_class(model_name):
        return getattr(sys.modules[__name__], model_name)


class ModelFactoryExtraTrees(ModelFactory):
    def __init__(self, config, datasets, model_type, cv_enable=None, n_estimators=100):
        super().__init__(config, datasets, model_type, 'ExtraTrees', cv_enable)
        self.model = None
        self.n_estimators = n_estimators

    def _fit(self):
        """ Create a ExtraTrees model """
        if self.is_regression():
            m = ModelSkExtraTreesRegressor(self.config, self.datasets, n_jobs=-1, n_estimators=self.n_estimators)
        elif self.is_classification():
            m = ModelSkExtraTreesClassifier(self.config, self.datasets, n_jobs=-1, n_estimators=self.n_estimators)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        if self.cv_enable is not None:
            m.cv_enable = self.cv_enable
        m.model_create()
        m.model_train()
        return m


class ModelFactoryGradientBoosting(ModelFactory):
    def __init__(self, config, datasets, model_type, cv_enable=None):
        super().__init__(config, datasets, model_type, 'GradientBoosting', cv_enable)
        self.model = None

    def _fit(self):
        """ Create a ExtraTrees model """
        if self.is_regression():
            m = ModelSkGradientBoostingRegressor(self.config, self.datasets)
        elif self.is_classification():
            m = ModelSkGradientBoostingClassifier(self.config, self.datasets)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        if self.cv_enable is not None:
            m.cv_enable = self.cv_enable
        m.model_create()
        m.model_train()
        return m


class ModelFactoryRandomForest(ModelFactory):
    def __init__(self, config, datasets, model_type, cv_enable=None, n_estimators=100, max_depth=None, bootstrap=True):
        super().__init__(config, datasets, model_type, 'RandomForestRegressor', cv_enable)
        self.model = None
        self.n_estimators, self.max_depth, self.bootstrap = n_estimators, max_depth, bootstrap

    def _fit(self):
        if self.is_regression():
            m = ModelSkRandomForestRegressor(self.config, self.datasets, n_jobs=-1, n_estimators=self.n_estimators, max_depth=self.max_depth, bootstrap=self.bootstrap)
        elif self.is_classification():
            m = ModelSkRandomForestClassifier(self.config, self.datasets, n_jobs=-1, n_estimators=self.n_estimators, max_depth=self.max_depth, class_weight='balanced', bootstrap=self.bootstrap)
        else:
            raise Exception(f"Unknown model type '{self.model_type}'")
        if self.cv_enable is not None:
            m.cv_enable = self.cv_enable
        m.model_create()
        m.model_train()
        return m
