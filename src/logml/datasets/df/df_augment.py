import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.decomposition import NMF, PCA

from ...core.config import CONFIG_DATASET_AUGMENT
from ...core.log import MlLog
from .df_normalize import DfNormalize
from .df_impute import DfImpute
from .methods_fields import FieldsParams


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
        self.pca_augment = None  # DfAugmentPca object
        if set_config:
            self._set_from_config()

    def augment(self, name, augmenter):
        """ Use an 'augmenter' object to add coluns to the dataFrame """
        ret = augmenter()
        if ret is None:
            self._debug("Augment dataframe: Could not do {name}")
            return False
        else:
            self.df = pd.concat([self.df, ret], axis=1)
            self._debug(f"Augment dataframe: DataFrame has shape {self.df.shape}, {name} result has shape {ret.shape}, joined datasets has shape {self.df.shape}")
            return True

    def __call__(self):
        """
        Augment dataframe
        Returns a new (augmented) dataset
        """
        if not self.enable:
            self._debug(f"Augment dataframe disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_AUGMENT}', enable='{self.enable}'")
            return self.df
        cols = "', '".join([c for c in self.df.columns])
        self._debug(f"Augment dataframe: Start. Shape: {self.df.shape}. Fields ({len(self.df.columns)}): ['{cols}']")
        self._op_add()
        self._op_sub()
        self._op_mult()
        self._op_div()
        self._op_log_ratio()
        self._op_logp1_ratio()
        self._nmf()
        self._pca()
        self._debug(f"Augment dataframe: Finished")
        cols = "', '".join([c for c in self.df.columns])
        self._debug(f"Augment dataframe: End. Shape: {self.df.shape}. Fields ({len(self.df.columns)}): ['{cols}']")
        return self.df

    def _op_add(self):
        self.add_augment = DfAugmentOpAdd(self.df, self.config, self.outputs, self.model_type)
        return self.augment('add', self.add_augment)

    def _op_div(self):
        self.divide_augment = DfAugmentOpDiv(self.df, self.config, self.outputs, self.model_type)
        return self.augment('div', self.divide_augment)

    def _op_mult(self):
        self.multiply_augment = DfAugmentOpMult(self.df, self.config, self.outputs, self.model_type)
        return self.augment('mult', self.multiply_augment)

    def _op_log_ratio(self):
        self.log_ratio_augment = DfAugmentOpLogRatio(self.df, self.config, self.outputs, self.model_type)
        return self.augment('log_ratio', self.log_ratio_augment)

    def _op_logp1_ratio(self):
        self.logp1_ratio_augment = DfAugmentOpLogPlusOneRatio(self.df, self.config, self.outputs, self.model_type)
        return self.augment('logp1_ratio', self.logp1_ratio_augment)

    def _op_sub(self):
        self.sub_augment = DfAugmentOpSub(self.df, self.config, self.outputs, self.model_type)
        return self.augment('sub', self.sub_augment)

    def _nmf(self):
        self.nmf_augment = DfAugmentNmf(self.df, self.config, self.outputs, self.model_type)
        return self.augment('NMF', self.nmf_augment)

    def _pca(self):
        self.pca_augment = DfAugmentPca(self.df, self.config, self.outputs, self.model_type)
        return self.augment('PCA', self.pca_augment)


class DfAugmentOp(FieldsParams):
    ''' Augment dataset by adding "operations" between fields '''

    def __init__(self, df, config, subsection, outputs, model_type, set_config=True):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, subsection, df.columns, outputs)
        self.operation_name = subsection
        self.symmetric = True

    def calc(self, namefieldparams, x):
        """Calculate the operation on pairwise fields from dataframe
        Returns: A dataframe of 'operations' (None on failure)
        """
        self._debug(f"Calculating {self.operation_name}: Start, name={namefieldparams.name}, params={namefieldparams.params}, fields:{namefieldparams.fields}")
        dfs = list()
        skip_second = set()
        for i in range(len(namefieldparams.fields)):
            field_i = namefieldparams.fields[i]
            if not self.can_apply_first(field_i):
                self._debug(f"Calculating {self.operation_name}: Cannot apply operation '{self.operation_name}' to first field '{field_i}', skipping")
                continue
            for j in range(len(namefieldparams.fields)):
                if i == j:
                    continue
                if not self.symmetric and i < j:
                    continue
                field_j = namefieldparams.fields[j]
                if field_j in skip_second:
                    continue
                if not self.can_apply_second(field_j):
                    self._debug(f"Calculating {self.operation_name}: Cannot apply operation '{self.operation_name}' to second field '{field_j}', skipping")
                    skip_second.add(field_j)
                    continue
                res = self.op(field_i, field_j)
                dfs.append(res)
        self._debug(f"Calculating {self.operation_name}: End")
        if len(dfs) > 0:
            df = pd.concat(dfs, axis=1)
            self._debug(f"Calculating {self.operation_name}: DataFrame joined shape {df.shape}")
            return df
        return None

    def can_apply_first(self, field):
        """ Can we apply the operation 'op(field_1, field_2) to the first fild 'field_1'? """
        return True

    def can_apply_second(self, field):
        """ Can we apply the operation 'op(field_1, field_2) to the second fild 'field_2'? """
        return True

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")


class DfAugmentOpAdd(DfAugmentOp):
    ''' Augment dataset by adding two fields '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, 'add', outputs, model_type, set_config=True)

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return self.df[field_i] + self.df[field_j]


class DfAugmentOpDiv(DfAugmentOp):
    ''' Augment dataset by dividing two fields '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, 'div', outputs, model_type, set_config=True)

    def can_apply_second(self, field):
        """ We apply this operation only if all numbers in the second field are non-zero """
        return (self.df[field] != 0).all()

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return self.df[field_i] / self.df[field_j]


class DfAugmentOpLogRatio(DfAugmentOp):
    ''' Augment dataset by applying the log ratio of two fields '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, 'log_ratio', outputs, model_type, set_config=True)

    def can_apply_first(self, field):
        """ We apply this operation if all numbers in the first field are positive """
        return (self.df[field] > 0).all()

    def can_apply_second(self, field):
        """ We apply this operation if all numbers in the second field are positive """
        return (self.df[field] > 0).all()

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return (self.df[field_i].log() - self.df[field_j].log()) / self.log_base


class DfAugmentOpLogPlusOneRatio(DfAugmentOp):
    ''' Augment dataset by applying the log+1 ratio of two fields '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, 'logp1_ratio', outputs, model_type, set_config=True)

    def can_apply_first(self, field):
        """ We apply this operation if all numbers in the first field are non-negative """
        return (self.df[field] >= 0).all()

    def can_apply_second(self, field):
        """ We apply this operation if all numbers in the second field are non-negative """
        return (self.df[field] >= 0).all()

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        n = (self.df[field_i] + 1).log()
        d = (self.df[field_j] + 1).log()
        return (n - d) / self.log_base


class DfAugmentOpMult(DfAugmentOp):
    ''' Augment dataset by multiplying two fields '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, 'mult', outputs, model_type, set_config=True)

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return self.df[field_i] * self.df[field_j]


class DfAugmentOpSub(DfAugmentOp):
    ''' Augment dataset by substracting two fields '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, 'sub', outputs, model_type, set_config=True)

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return self.df[field_i] - self.df[field_j]


class DfAugmentNmf(FieldsParams):
    ''' Augment dataset by adding Non-negative martix factorization '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, 'nmf', df.columns, outputs, params=['num'], madatory_params=['num'])
        self.sk_nmf_by_name = dict()

    def calc(self, namefieldparams, x):
        """Calculate 'num' NMFs using 'fields' from dataframe
        Returns: A dataframe of NMFs (None on failure)
        """
        self._debug(f"Calculating NMF: Start, name={namefieldparams.name}, num={namefieldparams.number}, fields:{namefieldparams.fields}")
        nmf = NMF(n_components=namefieldparams.number)
        nmf.fit(x)
        self.sk_nmf_by_name[namefieldparams.name] = nmf
        xnmf = nmf.transform(x)
        self._debug(f"Calculating NMF: End")
        return xnmf


class DfAugmentPca(FieldsParams):
    ''' Augment dataset by adding principal components '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, 'pca', df.columns, outputs, params=['num'], madatory_params=['num'])
        self.sk_pca_by_name = dict()

    def calc(self, namefieldparams, x):
        """Calculate 'num' PCAs using 'fields' from dataframe
        Returns: A dataframe of PCAs (None on failure)
        """
        self._debug(f"Calculating PCA: Start, name={namefieldparams.name}, num={namefieldparams.number}, fields:{namefieldparams.fields}")
        if x.isnull().sum().sum() > 0:
            cols_na = x.isnull().sum(axis=0) > 0
            # self._fatal_error(f"There are NA values i the dataset, columns: {cols_na}")
            self._fatal_error(f"Calculating PCA: There are NA values in the inputs, columns: {list(x.columns[cols_na])}")
        pca = PCA(n_components=namefieldparams.number)
        pca.fit(x)
        self.sk_pca_by_name[namefieldparams.name] = pca
        xpca = pca.transform(x)
        self._debug(f"Calculating PCA: End")
        return xpca
