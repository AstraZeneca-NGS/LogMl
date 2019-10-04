import numpy as np
import pandas as pd


class ResultsDf:
    def __init__(self, index):
        self.index = index
        self.df = pd.DataFrame({}, index=index)

    def add_col(self, name, vals):
        ''' Add column 'name:vals' to dataframe '''
        df_new = pd.DataFrame({name: vals}, index=self.index)
        self.df = self.df.join(df_new)

    def add_col_rank(self, name, vals, reversed=False):
        ''' Add a column ranked by value '''
        temp = vals.argsort()
        ranks = np.empty_like(temp)
        if reversed:
            temp = temp[::-1]
        ranks[temp] = np.arange(len(vals))
        self.add_col(name, ranks)

    def add_rank_of_ranksum(self):
        '''
        Add a column with the sum of all columns having 'rank' in the name.
        Also, add the rank of the previous column (i.e. rank of 'rank sum')
        '''
        len = self.df.shape[0]
        ranks = np.zeros(len)
        for c in self.df.columns:
            if 'rank' in c:
                ranks = ranks + self.df[c]
        self.add_col_rank("ranksum", ranks)
        self.add_col_rank("rank_of_ranksum", ranks)
