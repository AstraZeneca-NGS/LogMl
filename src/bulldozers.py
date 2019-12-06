#!/usr/bin/env python

#
# Example of MlDf (dataframe) using 'buldozers' dataset
#


import logging
import numpy as np
import os

from logml import *
from sklearn.ensemble import RandomForestRegressor


@dataset_inout
def df_in_out(df):
    " Split dataframe into inputs / output "
    return df.drop('SalePrice', axis=1), df.SalePrice


@dataset_preprocess
def df_preprocess(df):
    df.SalePrice = np.log(df.SalePrice)
    return df


@dataset_split
def df_split(df, n):
    return df[0:n], df[-n:], None


@model_evaluate
def meval(m, df):
    x, y = _in_out(df)
    return 1.0 - m.score(x, y)


@model_train
def mtrain(m, df):
    x, y = _in_out(df)
    return m.fit(x, y)


if __name__ == "__main__":
    ml = LogMl()
    ml()
    print("Done!")
