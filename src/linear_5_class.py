#!/usr/bin/env python

# Example of using 'linear 5 + class' dataset

# How dataset is built:
#
# # Number of samples
# num = 1000
# x1 = np.random.normal(0, 1, num)
# x2 = np.random.normal(2, 3, num)
# x3 = np.random.normal(-3, 5, num)
# x4 = np.random.normal(0, 1, num)
# c1 = (3 * np.random.rand(num)).astype(int)
# c1_classes = {0: 'low', 1:' mid', 2: 'high'}
# c1_str = np.array([c1_classes[c] for c in c1])
# c2 = (5 * np.random.rand(num)).astype(int)
# c2_classes = {0: 'very_low', 1: 'low', 2:' mid', 3: 'high', 4:'very_high'}
# c2_str = np.array([c2_classes[c] for c in c2])
# # Noise
# n = np.random.normal(0, 1, num)
# # Output
# y = 3. * x1 - 1. * x2 + 0.5 * x3 + 0.5 * c1 - 0.3 * c2 + 0.1 * n
# # DataFrame
# pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'c1': c1_str, 'c2': c2_str, 'y': y})tr})


from logml import *

ml = LogMl()
ml()
print("Done!")
