#!/usr/bin/env python

from logml import *

# Example of using 'Clasification: 3 inputs' dataset


def to_class(c):
    if c < -3:
        return 'low'
    if c < 3:
        return 'mid'
    return 'high'


# Create dataset
def create_dataset():
    # Number of samples
    num = 2000
    # Inputs: x1, x2, x3
    x1 = np.random.normal(0, 4, num)
    x2 = np.random.normal(2, 3, num)
    x3 = np.random.normal(-2, 3, num)
    # Noise
    n = np.random.normal(0, 1, num)
    # Output
    y = 3. * x1 - 1. * x2 + 0.5 * x3 + 0.1 * n
    # Categorical output
    y_str = np.array([to_class(c) for c in y])
    # Add missing data
    x1_na = (np.random.rand(num) < 0.01)
    x2_na = (np.random.rand(num) < 0.01)
    x3_na = (np.random.rand(num) < 0.01)
    x1[x1_na] = np.nan
    x2[x2_na] = np.nan
    x3[x3_na] = np.nan
    y_na = (np.random.rand(num) < 0.01)
    y[y_na] = np.nan
    # Create dataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y_str})
    df.to_csv('class3c.csv', index=False)


ml = LogMl()
ml()
print("Done!")
