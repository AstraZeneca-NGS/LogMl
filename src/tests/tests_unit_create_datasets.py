
import logging
import numpy as np
import pandas as pd
import os
import random
import sys
import time


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


# Create dataset
def create_dataset_transform_001():
    # Number of samples
    num = 1000
    # Inputs: x1, .. ., xn
    x1 = np.random.normal(0, 1, num)
    x2 = np.random.rand(num) * 5 + 7
    n = np.random.normal(0, 1, num)
    y = 3. * x1 - 1. * x2 + 0.1 * n
    y_na = (np.random.rand(num) < 0.1)
    y[y_na] = np.nan
    # Create dataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    file = 'test_dataset_transform_001.csv'
    print(f"Saving dataset to file '{file}'")
    df.to_csv(file, index=False)
    return df


def create_dataset_transform_002():
    num = 1000
    # Inputs: x1, .. ., xn
    x1 = np.random.normal(0, 1, num)
    x2 = np.random.rand(num) * 5 + 7
    d1 = np.array([rand_date() for _ in range(num)], dtype="datetime64[s]")
    n = np.random.normal(0, 1, num)
    y = 3. * x1 - 1. * x2 + 0.1 * n
    # Set some 'na'
    d1_na = (np.random.rand(num) < 0.1)
    d1[d1_na] = np.datetime64("NaT")
    # Create dataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'd1': d1, 'y': y})
    print(df.head())
    file = 'test_dataset_transform_002.csv'
    print(f"Saving dataset to file '{file}'")
    df.to_csv(file, index=False)
    return df


# Create dataset for PCA test
def create_dataset_pca(num=1000, prob_na=0.05):
    """
    Create a dataset of two input variables (independent random) and transform them using
        W = [[1,   0.5],
             [0.3, 0.5]]
    This should result in a covaraince matrix:
        W.T @ W = [[1.09, 0.65],
                   [0.65, 0.5 ]]
    And PCA components:
        np.linalg.eig(C) =
                [1.5088102, 0.0811898]          # Eigenvalues
                [[ 0.84061737, -0.54162943],    # Eigenvectors
                 [ 0.54162943,  0.84061737]]
    """
    dfdict = dict()
    # Inputs
    for i in range(2):
        dfdict[f"x{i}"] = rand_norm(num, 0, 1, prob_na)
    # Output
    dfdict['y'] = rand_norm(num, 0, 1, -1)
    # Create dataFrame
    df = pd.DataFrame(dfdict)
    # Covariates
    X = df[['x0', 'x1']].values
    W = np.array([[1, 0.5], [0.3, 0.5]])
    Xa = X @ W
    df[['x0', 'x1']] = Xa
    # Save to csv file
    df.to_csv('zzz.csv', index=False)
    return df


# Create dataset
def create_dataset_multiple_logistic_regression_pvalue():
    # Number of samples
    num = 1000
    # Inputs: x1, .. ., xn
    x1 = rand_norm(num)
    x2 = rand_norm(num)
    x3 = rand_norm(num)
    x4 = rand_norm(num)
    x5 = rand_norm(num)
    # Noise
    n = rand_norm(num)
    # Output
    y = 3. * x1 - 1. * x2 + 0.5 * x3 + 0.1 * n
    # Categorical output
    y_str = np.array([to_num(c) for c in y])
    # Create dataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'y': y_str})
    file = 'zzz.csv'
    print(f"Saving dataset to file '{file}'")
    df.to_csv(file, index=False)
    return df


def is_sorted(x):
    """ Is numpy array 'x' sorted? """
    return np.all(x[:-1] <= x[1:])


def logit(h):
    ''' Logistic from activation h '''
    p = 1.0 / (1.0 + np.exp(-h))
    r = np.random.rand(len(p))
    y = (r < p).astype('int')
    return y


def rand_date():
    max_time = int(time.time())
    t = random.randint(0, max_time)
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def rand_unif(num, mean, std, na_prob=0):
    xi = np.random.rand(num)
    if na_prob > 0:
        xi_na = (np.random.rand(num) <= na_prob)
        xi[xi_na] = np.nan
    return xi


def rand_norm(num, mean=0.0, std=1.0, na_prob=0):
    xi = np.random.normal(mean, std, num)
    if na_prob > 0:
        xi_na = (np.random.rand(num) <= na_prob)
        xi[xi_na] = np.nan
    return xi


def rand_choice(num, val_max, na_prob=0):
    xi = 1.0 * np.random.choice(val_max, size=num)
    if na_prob > 0:
        xi_na = (np.random.rand(num) < 0.1)
        xi[xi_na] = np.nan
    return xi


def to_class(c):
    if c < -3:
        return 'low'
    if c < 3:
        return 'mid'
    return 'high'


def to_num(c):
    """ Convert to a class, encoded as number
    {'low', 'mid', 'high'} => {0, 1, 2}
    """
    if c < -3:
        return 0
    if c < 3:
        return 1
    return 2
