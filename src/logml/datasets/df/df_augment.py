import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.decomposition import NMF, PCA

from ...core.config import CONFIG_DATASET_AUGMENT
from ...core.log import MlLog
from ...util.counter_dim import CounterDimIncreasing
from .df_normalize import DfNormalize
from .methods_fields import FieldsParams


DEFAULT_ORDER = 2


def _parse_base(base):
    """ Initialize operations """
    if base is None:
        return np.log(2)
    elif base == 'e':
        return 1.0
    else:
        return np.log(base)


class DfAugment(MlLog):
    '''
    DataFrame augmentation
    '''

    def __init__(self, df, config, outputs, model_type, set_config=True):
        super().__init__(config, CONFIG_DATASET_AUGMENT)
        self.config = config
        self.df = df
        self.dfs = list()
        self.outputs = outputs
        self.model_type = model_type
        if set_config:
            self._set_from_config()

    def augment(self, name, augmenter):
        """ Use an 'augmenter' object to add coluns to the dataFrame """
        ret = augmenter()
        if ret is None:
            self._debug(f"Augment dataframe: Could not do {name}")
            return False
        else:
            self.dfs.append(ret)
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
        self._info(f"Augment dataframe: Start. Shape: {self.df.shape}. Fields ({len(self.df.columns)}): ['{cols}']")
        self._op_add()
        self._op_and()
        self._op_div()
        self._op_log_ratio()
        self._op_logp1_ratio()
        self._op_mult()
        self._op_sub()
        self._nmf()
        self._pca()
        if len(self.dfs) > 0:
            to_join = [self.df]
            to_join.extend(self.dfs)
            df_joined = pd.concat(to_join, axis=1)
            self._info(f"Augment dataframe: DataFrame has shape {self.df.shape}, results have shapes {[ret.shape for ret in self.dfs]}")
            self.df = df_joined
            cols = "', '".join([c for c in self.df.columns])
            self._info(f"Augment dataframe: End. Shape: {self.df.shape}. Fields ({len(self.df.columns)}): ['{cols}']")
        return self.df

    def __getstate__(self):
        " Do not pickle any '_augment' objects "
        state = self.__dict__.copy()
        to_delete = [k for k in state.keys() if k.endswith('_augment')]
        for k in to_delete:
            del state[k]
        return state

    def _op_add(self):
        self.add_augment = DfAugmentOpAdd(self.df, self.config, self.outputs, self.model_type)
        return self.augment('add', self.add_augment)

    def _op_and(self):
        self.and_augment = DfAugmentOpAnd(self.df, self.config, self.outputs, self.model_type)
        return self.augment('and', self.and_augment)

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


class DfAugmentOpBinary(FieldsParams):
    '''
    Augment dataset by adding "binary operations" between two fields (e.g. difference,
    ratio, log ratio, etc.)
    '''

    def __init__(self, df, config, subsection, outputs, model_type, params=None, madatory_params=None):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, subsection, df.columns, outputs, params, madatory_params)
        self.operation_name = subsection
        self.symmetric = True
        self.min_non_zero_count = 1
        self.order = 2

    def calc(self, namefieldparams, x):
        """Calculate the operation on pairwise fields from dataframe
        Returns: A dataframe of 'operations' (None on failure)
        """
        self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Start, name={namefieldparams.name}, params={namefieldparams.params}, fields:{namefieldparams.fields}")
        results = list()
        skip_second = set()
        self._op_init(namefieldparams)
        cols = list()
        for i in range(len(namefieldparams.fields)):
            field_i = namefieldparams.fields[i]
            if not self.can_apply_first(field_i):
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Cannot apply operation '{self.operation_name}' to first field '{field_i}', skipping")
                continue
            for j in range(len(namefieldparams.fields)):
                if i == j:
                    continue
                if self.symmetric and i > j:
                    continue
                field_j = namefieldparams.fields[j]
                if field_j in skip_second:
                    continue
                if not self.can_apply_second(field_j):
                    self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Cannot apply operation '{self.operation_name}' to second field '{field_j}', skipping")
                    skip_second.add(field_j)
                    continue
                res = self.op(field_i, field_j)
                if self.should_add(field_i, field_j, res):
                    cols.append(f"{namefieldparams.name}_{field_i}_{field_j}")
                    results.append(res)
                else:
                    self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Should add returned False, skipping")
        self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': End")
        if len(results) > 0:
            x = np.concatenate(results)
            x = x.reshape(-1, len(results))
            df = self.array_to_df(x, cols)
            self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': DataFrame joined shape {df.shape}")
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

    def _op_init(self, namefieldparams):
        """ Initialize operations """
        min_non_zero = namefieldparams.params.get('min_non_zero')
        if min_non_zero is not None:
            if min_non_zero < 0.0:
                self._error(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Illegal value for 'min_non_zero'={min_non_zero}, ignoring")
            if 0.0 < min_non_zero and min_non_zero < 1.0:
                self.min_non_zero_count = int(min_non_zero * len(self.df))
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Setting min_non_zero_count={self.min_non_zero_count}, min_non_zero: {min_non_zero}, len(df): {len(self.df)}")
            else:
                self.min_non_zero_count = int(min_non_zero)
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Setting min_non_zero_count={self.min_non_zero_count}")

    def should_add(self, field_i, field_j, x):
        """ Should we add add these results """
        count_non_zero = (x != 0.0).sum()
        ok = count_non_zero >= self.min_non_zero_count
        if not ok:
            self._debug(f"Calculating {self.operation_name}, fields {field_i}, {field_j}: Minimum number of non-zero fields is {self.min_non_zero_count}, but there are only {count_non_zero} non-zeros. Not adding column")
        return ok


class DfAugmentOpNary(FieldsParams):
    '''
    Augment dataset by adding "N-ary operations" between (two or more) fields (e.g. sum)
    '''

    def __init__(self, df, config, subsection, outputs, model_type, params=None, madatory_params=None):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, subsection, df.columns, outputs, params, madatory_params)
        self.operation_name = subsection
        self.min_non_zero_count = 1
        self.order = DEFAULT_ORDER

    def calc(self, namefieldparams, x):
        """
        Calculate the operation on N fields from dataframe
        Returns: A dataframe of 'operations' (None on failure)
        """
        self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Start, name={namefieldparams.name}, params={namefieldparams.params}, fields:{namefieldparams.fields}")
        results = list()
        self._op_init(namefieldparams)
        cols = list()
        counter = CounterDimIncreasing(len(namefieldparams.fields), self.order)
        count, count_added, count_skipped = 1, 0, 0
        for nums in counter:
            fields = [namefieldparams.fields[i] for i in nums]
            res = self.op(fields)
            count += 1
            if self.should_add(fields, res):
                count_added += 1
                cols.append(f"{namefieldparams.name}_{'_'.join(fields)}")
                results.append(res)
            else:
                count_skipped += 1
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Fields {fields}. Should add returned False, skipping")
            if count % 100 == 0:
                self._info(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Minimum non-zero: {self.min_non_zero_count}, count {count}, added {count_added}, skipped {count_skipped}")
        self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': End")
        if len(results) > 0:
            x = np.concatenate(results)
            x = x.reshape(-1, len(results))
            df = self.array_to_df(x, cols)
            self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': DataFrame joined shape {df.shape}")
            return df
        return None

    def op(self, fields):
        """ Calculate the arithmetic operation between the two fields """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def _op_init(self, namefieldparams):
        """ Initialize operations """
        min_non_zero = namefieldparams.params.get('min_non_zero')
        if min_non_zero is not None:
            if min_non_zero < 0.0:
                self._error(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Illegal value for 'min_non_zero'={min_non_zero}, ignoring")
            if 0.0 < min_non_zero and min_non_zero < 1.0:
                self.min_non_zero_count = int(min_non_zero * len(self.df))
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Setting min_non_zero_count={self.min_non_zero_count}, min_non_zero: {min_non_zero}, len(df): {len(self.df)}")
            else:
                self.min_non_zero_count = int(min_non_zero)
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Setting min_non_zero_count={self.min_non_zero_count}")
        self.order = namefieldparams.params.get('order')
        if self.order is None:
            self.order = DEFAULT_ORDER

    def should_add(self, fields, x):
        """ Should we add add these results """
        count_non_zero = (x != 0.0).sum()
        ok = count_non_zero >= self.min_non_zero_count
        if not ok:
            self._debug(f"Calculating {self.operation_name}, fields {fields}: Minimum number of non-zero fields is {self.min_non_zero_count}, but there are only {count_non_zero} non-zeros. Not adding column")
        return ok


class DfAugmentOpNaryIncremental(FieldsParams):
    '''
    Augment dataset by adding "N-ary operations" between (two or more) fields (e.g. 'and') in an incremental way
    This is much more efficient when there are many new values not incorporated due to having too many zeros.
    '''

    def __init__(self, df, config, subsection, outputs, model_type, params=None, madatory_params=None):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, subsection, df.columns, outputs, params, madatory_params)
        self.operation_name = subsection
        self.min_non_zero_count = 1
        self.order = DEFAULT_ORDER

    def calc(self, namefieldparams, x):
        """
        Calculate the operation on N fields from dataframe
        Returns: A dataframe of 'operations' (None on failure)
        """
        self._info(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Start, name={namefieldparams.name}, params={namefieldparams.params}, fields:{namefieldparams.fields}")
        self._op_init(namefieldparams)
        aug_dict = self.incremental_init(namefieldparams)
        if self.order < 2:
            self._error(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Order is less than 2 (order='{self.order}')")
            return None
        for i in range(1, self.order):
            self.incremental(namefieldparams)
        df = self.get_results(namefieldparams)
        self._info(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': End, dataframe.shape: {df.shape if df is not None else '-'}")
        return df

    def convert_result(self, x):
        return x.astype('float')

    def field_nums_to_name(self, namefieldparams, field_nums):
        """ Convert a list of field numbers to field names """
        fields = [namefieldparams.fields[fn] for fn in field_nums]
        return f"{namefieldparams.name}_{'_'.join(fields)}"

    def get_augment_key(self, fieldnum_list):
        """ Get key for augment dictionary (indexed by field numbers) """
        return '\t'.join([str(i) for i in fieldnum_list])

    def get_results(self, namefieldparams):
        """ Get all results frmo self.augment dictionary into a dataframe """
        results, col_names = list(), list()
        # Add all items in dict, create a list of column names
        keys = sorted(list(self.augment.keys()))
        for key in keys:
            x = self.convert_result(self.augment[key])
            results.append(x)
            col_name = self.field_nums_to_name(namefieldparams, self.augment_fieldnums[key])
            col_names.append(col_name)
        # Merge into a dataframe
        if len(results) > 0:
            df = pd.concat(results, axis=1)
            df.columns = col_names
            self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': DataFrame joined shape {df.shape}")
            return df
        return None

    def incremental_init(self, namefieldparams):
        """ Initialize incremental dictionaries indexed by field numbers (as strings) """
        self.augment = dict()
        self.augment_fieldnums = dict()
        for fnum in range(len(namefieldparams.fields)):
            field = namefieldparams.fields[fnum]
            self.set_augment([fnum], self.init(field))

    def incremental(self, namefieldparams):
        """ Incrementally add one field at a time """
        augment = dict()
        augment_fieldnums = dict()
        count, count_added, count_skipped = 1, 0, 0
        for key in self.augment.keys():
            value = self.augment[key]
            fnlist = self.augment_fieldnums[key]
            fnmax = max(fnlist)
            fields = [namefieldparams.fields[fn] for fn in fnlist]
            # Iterate over all fields we can add. We don't want to repeat fields, so number of fields is always incrementing
            for fnum in range(fnmax + 1, len(namefieldparams.fields)):
                count += 1
                field = namefieldparams.fields[fnum]
                x = self.op_inc(value, field)
                if self.should_add(fields, field, x):
                    # Add to new entry (old list of field numbers + new field number)
                    count_added += 1
                    fnlist_new = fnlist.copy()
                    fnlist_new.append(fnum)
                    key = self.get_augment_key(fnlist_new)
                    augment[key] = x
                    augment_fieldnums[key] = fnlist_new
                else:
                    count_skipped += 1
                if count % 100 == 0:
                    self._info(f"Incremental calculation {self.operation_name}, name: '{namefieldparams.name}': Minimum non-zero: {self.min_non_zero_count}, count {count}, added {count_added}, skipped {count_skipped}, fields {fields} + {field}")
        # Update incremental values
        self.augment = augment
        self.augment_fieldnums = augment_fieldnums

    def init(self, field):
        return self.df[field]

    def op(self, fields):
        """
        Calculate the arithmetic operation between the two fields
        @returns A Pandas Series
        """
        raise NotImplementedError("Unimplemented method, this method should be overiden by a subclass!")

    def _op_init(self, namefieldparams):
        """ Initialize operations """
        min_non_zero = namefieldparams.params.get('min_non_zero')
        if min_non_zero is not None:
            if min_non_zero < 0.0:
                self._error(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Illegal value for 'min_non_zero'={min_non_zero}, ignoring")
            if 0.0 < min_non_zero and min_non_zero < 1.0:
                self.min_non_zero_count = int(min_non_zero * len(self.df))
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Setting min_non_zero_count={self.min_non_zero_count}, min_non_zero: {min_non_zero}, len(df): {len(self.df)}")
            else:
                self.min_non_zero_count = int(min_non_zero)
                self._debug(f"Calculating {self.operation_name}, name: '{namefieldparams.name}': Setting min_non_zero_count={self.min_non_zero_count}")
        self.order = namefieldparams.params.get('order')
        if self.order is None:
            self.order = DEFAULT_ORDER

    def set_augment(self, fieldnum_list, value):
        """ Set an entry in augment dictionaty (indexed by field numbers) """
        key = self.get_augment_key(fieldnum_list)
        self.augment[key] = value
        self.augment_fieldnums[key] = fieldnum_list

    def should_add(self, fields, field, x):
        """ Should we add add these results """
        count_non_zero = (x != 0.0).sum()
        ok = count_non_zero >= self.min_non_zero_count
        if ok:
            self._debug(f"Calculating {self.operation_name}, fields {fields} + {field}: Number of non-zero fields {count_non_zero} >= {self.min_non_zero_count}, OK")
        else:
            self._debug(f"Calculating {self.operation_name}, fields {fields} + {field}: Minimum number of non-zero fields is {self.min_non_zero_count}, but there are only {count_non_zero} non-zeros. Not adding column")
        return ok


class DfAugmentOpAdd(DfAugmentOpNary):
    ''' Augment dataset by adding two or more fields '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, 'add', outputs, model_type, params=['min_non_zero', 'order'])

    def op(self, fields):
        """ Calculate the arithmetic operation between the two or more fields """
        return self.df[fields].sum(axis=1)


class DfAugmentOpAnd(DfAugmentOpNaryIncremental):
    ''' Augment dataset by performing 'and' of two or more fields '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, 'and', outputs, model_type, params=['min_non_zero', 'order', 'threshold'])

    def init(self, field):
        return self.df[field] > self.threshold

    def _op_init(self, namefieldparams):
        super()._op_init(namefieldparams)
        self.threshold = namefieldparams.params.get('threshold')
        if self.threshold is None:
            self.threshold = 0.0

    def op_inc(self, x, field):
        """ Calculate the (incremental) 'and' operation between the a value and a field """
        return x & (self.df[field] > self.threshold)


class DfAugmentOpDiv(DfAugmentOpBinary):
    ''' Augment dataset by dividing two fields '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, 'div', outputs, model_type, params=['min_non_zero'])

    def can_apply_second(self, field):
        """ We apply this operation only if all numbers in the second field are non-zero """
        return (self.df[field] != 0).all()

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return self.df[field_i] / self.df[field_j]


class DfAugmentOpLogRatio(DfAugmentOpBinary):
    ''' Augment dataset by applying the log ratio of two fields '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, 'log_ratio', outputs, model_type, params=['base', 'min_non_zero'])

    def can_apply_first(self, field):
        """ We apply this operation if all numbers in the first field are positive """
        return (self.df[field] > 0).all()

    def can_apply_second(self, field):
        """ We apply this operation if all numbers in the second field are positive """
        return (self.df[field] > 0).all()

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return (np.log(self.df[field_i]) - np.log(self.df[field_j])) / self.log_base

    def _op_init(self, namefieldparams):
        """ Initialize operations """
        super()._op_init(namefieldparams)
        base = namefieldparams.params.get('base')
        self.log_base = _parse_base(base)
        self._debug(f"Log base is '{base}', setting log_base={self.log_base}")


class DfAugmentOpLogPlusOneRatio(DfAugmentOpBinary):
    ''' Augment dataset by applying the log+1 ratio of two fields '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, 'logp1_ratio', outputs, model_type, params=['base', 'min_non_zero'])

    def can_apply_first(self, field):
        """ We apply this operation if all numbers in the first field are non-negative """
        return (self.df[field] >= 0).all()

    def can_apply_second(self, field):
        """ We apply this operation if all numbers in the second field are non-negative """
        return (self.df[field] >= 0).all()

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        n = np.log(self.df[field_i] + 1)
        d = np.log(self.df[field_j] + 1)
        return (n - d) / self.log_base

    def _op_init(self, namefieldparams):
        """ Initialize operations """
        super()._op_init(namefieldparams)
        base = namefieldparams.params.get('base')
        self.log_base = _parse_base(base)
        self._debug(f"Log base is '{base}', setting log_base={self.log_base}")


class DfAugmentOpMult(DfAugmentOpNary):
    ''' Augment dataset by multiplying two or more fields '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, 'mult', outputs, model_type, params=['min_non_zero', 'order'])

    def op(self, fields):
        """ Calculate the arithmetic operation between the two or more fields """
        return self.df[fields].prod(axis=1)


class DfAugmentOpSub(DfAugmentOpBinary):
    ''' Augment dataset by substracting two fields '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, 'sub', outputs, model_type, params=['min_non_zero'])

    def op(self, field_i, field_j):
        """ Calculate the arithmetic operation between the two fields """
        return self.df[field_i] - self.df[field_j]


class DfAugmentNmf(FieldsParams):
    ''' Augment dataset by adding Non-negative martix factorization '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, 'nmf', df.columns, outputs, params=['min_non_zero', 'num'], madatory_params=['num'])
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
        cols = self.get_col_names_num(namefieldparams, xnmf)
        return self.array_to_df(xnmf, cols)


class DfAugmentPca(FieldsParams):
    ''' Augment dataset by adding principal components '''

    def __init__(self, df, config, outputs, model_type):
        super().__init__(df, config, CONFIG_DATASET_AUGMENT, 'pca', df.columns, outputs, params=['num'], madatory_params=['num'])
        self.sk_pca_by_name = dict()

    def calc(self, namefieldparams, x):
        """Calculate 'num' PCAs using 'fields' from dataframe
        Returns: A dataframe of PCAs (None on failure)
        """
        num = namefieldparams.params['num']
        self._debug(f"Calculating PCA: Start, name={namefieldparams.name}, num={num}, fields:{namefieldparams.fields}")
        if x.isnull().sum().sum() > 0:
            cols_na = x.isnull().sum(axis=0) > 0
            self._fatal_error(f"Calculating PCA: There are NA values in the inputs, columns: {list(x.columns[cols_na])}")
        pca = PCA(n_components=num)
        pca.fit(x)
        self.sk_pca_by_name[namefieldparams.name] = pca
        xpca = pca.transform(x)
        self._debug(f"Calculating PCA: End")
        cols = self.get_col_names_num(namefieldparams, xpca)
        return self.array_to_df(xpca, cols)
