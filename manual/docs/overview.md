
# Overview

`Log(ML)` is a Machine Learning / AI automation framework.
Here is a summary of what `LogMl` does:

1. Dataset:
	1. Load: Load a dataset from a file
	1. Transform: sanitize input names, encode cathegorical inputs, one-hot encoding, expand date/time fieds, remove duplicated inputs, remove samples with missing outputs, shuffle samples, remove inputs with low variance, impute missing values, add missing value indicator variables, normalize inputs, etc.
	1. Augment: add pricipal components (PCA), NMF, interaction variables, etc.
	1. Explore: show dataset, correlation analysis, dendogram, missingness analysis, normality/log-normality analysis, inputs distributions, pair-plots, heatmaps, etc.
	1. Split: Split data into train / validate / test datasets
	1. Inputs/Outputs: Obtain inputs and output for each (train / validate / test) dataset.
1. Feature importance:
	1. Model-based (Random Forest, Extra trees, Gradient Boosting) by input shuffle / drop column algorithms
	1. Boruta
	1. Regularization methods (Lasso, Ridge, Lars)
	1. Recursive Feature elimination (Lasso, Ridge, Lars AIC, Lars BIC, Random Forest, Extra trees, Gradient Boosting)
	1. Linear model p-value
	1. Logistic model p-value (Wilks)
	1. Multi-class Logistic model p-value (Wilks)
	1. Summary by weighted consensus of all the previous method using model's validation score.
1. Model Training:
	1. Single model: Train a model, calculate validation and test dataset performance
	1. Hyper-parameter optimization: Train models using hyper-parameter optimization using Bayesian or Random algorithms
1. Model Search: Search over 50 different model families (i.e. machine learning algorithms). This search can be done using hyper-parameter optimizations on the model's hyper-parameters.
1. Cross-validation: Perform cross-validation (KFold, RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit). Cross-validation can be used in all model search and Feature importance steps.
1. Logging: All outputs are directed to log files
1. Save models & datasets: Automatically save models and datasets for easier retrieval later
