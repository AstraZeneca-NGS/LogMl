import numpy as np
import pandas as pd

from IPython.core.display import Image, display


class ResultsDf:
    def __init__(self, index=None):
        self.index = index
        self.df = None
        if index is not None:
            self.df = pd.DataFrame({}, index=index)

    def add_df(self, df):
        ''' Add columns from dataframe 'df' to dataframe '''
        if self.df is None:
            self.df = df
        else:
            self.df = self.df.join(df)

    def add_col(self, name, vals):
        ''' Add column 'name:vals' to dataframe '''
        vals = self._flatten(vals)
        df_new = pd.DataFrame({name: vals}, index=self.index)
        self.df = self.df.join(df_new)

    def add_row(self, row_name, vals_dict):
        ''' Add a row of values '''
        df_row = pd.DataFrame(vals_dict, index=[row_name])
        self.add_row_df(df_row)

    def add_row_df(self, df):
        ''' Add (concatenate) a rows in 'df' '''
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df], sort=False)

    def _flatten(self, x):
        return x if x.ndim == 1 else x.flatten()

    def print(self, msg=None):
        if msg is not None:
            print(msg)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(self.df)

    def sort(self, col_names):
        if self.df is None:
            return
        self.df.sort_values(col_names, inplace=True)


class ResultsRankDf(ResultsDf):
    """ A resutls data frame with rankde values """

    def __init__(self, index=None):
        super().__init__(index)
        self.weights = dict()
        self.weight_default = 1.0

    def add_col_rank(self, name, vals, weight=None, reversed=False):
        ''' Add a column ranked by value '''
        vals = self._flatten(vals)
        s = pd.Series(vals, self.index)
        ranks = s.rank(ascending=not reversed, na_option='bottom')
        self.add_col(name, ranks)
        self.add_weight(name, weight)

    def add_rank_of_ranksum(self):
        '''
        Add a (weighted) column with the sum of all columns having 'rank' in the name.
        Also, add the rank of the previous column (i.e. rank of 'rank sum')
        '''
        len = self.df.shape[0]
        ranks_sum = np.zeros(len)
        for c in self.df.columns:
            if 'rank' in c:
                if c not in self.weights:
                    self.weights[c] = self.weight_default
                ranks_sum = ranks_sum + self.weights[c] * self.df[c]
        self.add_col("ranks_sum", ranks_sum)
        self.add_col_rank("rank_of_ranksum", ranks_sum)

    def add_weight(self, name, weight):
        if weight is not None:
            if weight <= 0.0:
                raise ValueError(f"Weight should be a positive number, got {weight}")
            self.weights[name] = weight

    def get_weights_table(self):
        names = list(self.weights.keys())
        names.sort()
        weights = np.array([self.weights.get(n, self.weight_default) for n in names])
        wt = ResultsDf(names)
        wt.add_col('weights', weights)
        return wt
