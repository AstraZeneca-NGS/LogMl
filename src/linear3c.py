#!/usr/bin/env python

from logml import *

# Example of using 'linear 3 input + 2 categorical' dataset


# Create dataset
def create_dataset():
    # Number of samples
    num = 1000
    # Inputs: x1, x2, x3
    x1 = np.random.normal(0, 4, num)
    x2 = np.random.normal(2, 3, num)
    x3 = np.random.normal(-2, 3, num)
    # Categorical input: c1
    c1 = (3 * np.random.rand(num)).astype(int)
    c1_classes = {0: 'low', 1: 'mid', 2: 'high'}
    c1_str = np.array([c1_classes[c] for c in c1])
    # Categorical input: c2
    c2 = (5 * np.random.rand(num)).astype(int)
    c2_classes = {0: 'very_low', 1: 'low', 2: 'mid', 3: 'high', 4: 'very_high'}
    c2_str = np.array([c2_classes[c] for c in c2])
    # Noise
    n = np.random.normal(0, 1, num)
    # Output
    y = 3. * x1 - 2. * x2 + 0.5 * x3 + 0.5 * c1 - 0.3 * c2 + 0.1 * n
    # Add missing data
    x1_na = (np.random.rand(num) < 0.01)
    x2_na = (np.random.rand(num) < 0.01)
    x3_na = (np.random.rand(num) < 0.01)
    x1[x1_na] = np.nan
    x2[x2_na] = np.nan
    x3[x3_na] = np.nan
    # Create dataFrame
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'c1': c1_str, 'c2': c2_str, 'y': y})
    df.to_csv('linear3c.csv', index=False)


ml = LogMl()
ml()
print("Done!")
