#!/usr/bin/env python

from logml import *

# Example of using linear 100 input
# In this dataset, we have several duplicated, random and colinear inputs

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
    # Output: Only dependent variables + noise
    y = 3. * x1 - 2. * x2 + 0.5 * x3 + 0.5 * c1 - 0.3 * c2 + 0.1 * n
    df_dict = {'x1': x1, 'x2': x2, 'x3': x3, 'c1': c1_str, 'c2': c2_str, 'y': y}
    # Add random inputs
    for i in range(40):
        mean = 10 * np.random.rand(1) - 5
        std = 5 * np.random.rand(1)
        xi = np.random.normal(mean, std, num)
        df_dict[f"rand_{i}"] = xi
    # Add duplicated columns
    cols_ori = [x1, x2, x3, c1, c2]
    for i in range(2 * len(cols_ori)):
        ci = i % len(cols_ori)
        df_dict[f"dup_{i}"] = cols_ori[ci]
    # Add colinear: Multiplier + noise
    cols_x = [x1, x2, x3]
    for i in range(len(cols_x) * 15):
        ci = i % len(cols_x)
        ki = np.random.normal(0, 1, 1)
        ni = np.random.normal(0, 1, num)
        df_dict[f"colin_{i}_x{ci}"] = ki * cols_x[ci] + 0.1 * ni
    # Create dataFrame
    df = pd.DataFrame(df_dict)
    df.to_csv('linear100c.csv', index=False)
    return df


ml = LogMl()
ml()
print("Done!")
