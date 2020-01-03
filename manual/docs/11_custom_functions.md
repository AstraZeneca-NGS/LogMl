
# AI/ML Workflow

`Log(ML)` performs the following series of steps (all of them customizable using Python functions and YAML configuration). `Log(ML)` allows you to define your own custom functions by adding annotations.

Here is a summary of the workflow steps (details are covered in the next sub-sections):

1. Dataset: Load or Create, Transform, Preprocess, Augment, Explore, Split, Inputs/Outputs
1. Feature importance
1. Model Training
	1. Cross-validation
	1. Hyper-parameter optimization
1. Model Search

Each section can be enabled / disabled and customized in the YAML configuration file.

# Workflow Overview

The main workflow in `Log(ML)` has the following steps (and their respective annotations):

1. Dataset
    1. Load: `@dataset_load`
    1. Create (if not loaded): `@dataset_create`
    1. Transform: `@dataset_transform`
    1. Augment: `@dataset_augment`
    1. Preprocess: `@dataset_preprocess`
    1. Split: `@dataset_split`
    1. Save: `@dataset_save`
1. Dataset
    1. Basic feature statistics
    1. Feature co-linearity analysis
    1. Feature importance
1. Model
    1. Create: `@model_create`
    1. Train: `@model_train`
    1. Save: `@model_save`
    1. Save train results
    1. Test: `@model_evaluate`
    1. Validate: `@model_evaluate`

# Main workflow: Dataset

This step takes care of loading or creating a dataset. There are several sub-steps taking care of different aspects of dataset processing to make sure is suitable for training a model

Here is an overview of "dataset" workflow is organized. Each sub-section below shows details for specific steps:

1. Load: `@dataset_load`
1. Create (if not loaded): `@dataset_create`
1. Transform: `@dataset_transform`
1. Augment: `@dataset_augment`
1. Preprocess: `@dataset_preprocess`
1. Split: `@dataset_split`
1. Save: `@dataset_save`

### YAML config: Dataset section

The config YAML file section for dataset part of the workflow is:
```
dataset:
  dataset_name: 'my_dataset'       # Dataset name
  dataset_path: 'data/my_dataset'  # Path to use when loading and saving datasets
  dataset_type: None               # Dataset type: 'df' means dataFrame (if this option is commented out then a custom object is assumed)
  is_use_default_split: False      # Use (internal) 'split' function if none is provided by the user?
```

Other options specific to DataFrames (i.e. `dataset_type: 'df'`):
```
dataset:
  dataset_type: 'df'
  ...
  categories:                                      # Define categorical data columns
    category_name_1: ['Small', 'Medium', 'Large']  # Force this column to be converted to a catergory, set categories as ['Small', 'Medium', 'Large'] and make sure the category is ordered in the same order
    category_name_2:                               # Force this column to be converted to a category (no particular order)
  dates: ['record_date']                           # Force these columns to be treated as a date_time, expand all date sub-fields into different columns (e.g. 'yyyy', 'mm', 'dd', 'day_of_week'... etc.)
  ont_hot: ['Enclosure_Type']                      # All columns listed here are converted to one hot encoding
  one_hot_max_cardinality: 7                       # All columns having less then this number of categories are converted to one hot encoding
  std_threshold: 0.0                               # Drop columns of having standard deviation less or equal than this threshold
```

### Dataset: Load
This step typical attempts to load data from files (e.g. load a "data frame" or a set of images).

1. Attempt to load from pickle file (`{dataset_path}/{dataset_name}.pkl`). If the files exists, load the dataset.
1. Invoke a user defined function decorated with `@dataset_load`.
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, so `Log(ML)` will attempt to create a dataset (next step)
    - Parameters to the user defined function are defined in config YAML file section `functions`, sub-section `dataset_load`
    - The dataset is marked to be saved

### Dataset: Create
This part creates a dataset by invoking the user defined function decorated by `@dataset_create`
1. If the dataset has been loaded (i.e. "Load dataset" step was successful), skip this step
1. Invoke a user defined function decorated with `@dataset_create`:
    - If there is no user defined function or the section is disabled in the config file (i.e. `enable=False`), this step fails. Since `Log(ML)` doesn't have a dataset to work with (load and create steps both failed) it will exit with an error.
    - Parameters to the user defined function are defined in config YAML file section `functions`, sub-section `dataset_create`
    - The return value from the user defined function is used as a dataset
    - The dataset is marked to be saved

**DataFrames:** Dataset load default implementation for `dataset_type='df'` (i.e. DataFrame) reads a dataFrame from a CSV file using `pandas.read_csv`

### Dataset: Transform
This step is used to make changes to dataset to make is usable, for instance converting string values into numerical categories.
1. If this step has already been performed (i.e. a dataset loaded from a pickle file that has already been transformed), the step is skipped
1. Invoke a user defined function decorated with `@dataset_transform`:
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, no error is produced.
    - Parameters: The first parameter is the dataset. Other parameters are defined in config YAML file section `functions`, sub-section `dataset_transform`
    - The return value replaces the original dataset
    - The dataset is marked to be saved

**DataFrames:** Dataset transform default implementation for `dataset_type='df'` (i.e. DataFrame)

<!-- 1. Sanitize variables names
1. Convert categorical variables
1. Expand date/time variables
1. perform one-hot encoding
1. Remove missing remove missing output
1. Remove columns (predefined, low standard deviation)
1. Augment: Dataset augmentation
1. Data Explore:
1. Summaries, normality, missing values, histograms, etc.
1. Correlation heatmap, top / bottom rank correlated, dendograms, etc.
1. Nullity analysis -->


<!-- 1. Summaries, normality, missing values, histograms, etc.
1. Correlation heatmap, top / bottom rank correlated, dendograms, etc.
1. Nullity analysis
1. Feature importance:
1. Model-based column permutation: Random Forest, ExtraTrees, GradientBoosting
1. Boruta algorithm
1. Model-based drop-column: Random Forest, ExtraTrees, GradientBoosting
1. Chi^2
1. Mutual information
1. SkLearn importance: Random Forest, ExtraTrees, GradientBoosting
1. Regularization methods: Lasso, Ridge, Lars (AIC), Lars (BIC)
1. Recursive Feature Elimination
1. Tree graph -->


1. Expand date/time features
1. Convert to categorical
1. Convert to one-hot
1. Missing data
1. Drop low standard deviation fields

**Expand date/time features**:

Fields defined in the config YAML file, section `dataset`, sub-section `dates` are treated as date/time when the dataFrame CSV is loaded and then expanded into several columns: `[Year, Month, Day, DayOfWeek, Hour, Minute, Second, etc.]`.

**Convert to categorical**:

Fields defined in the config YAML file, section `dataset`, sub-section `categories` are converted into categorical data and converted to a numerical (integer) representation. Category `-1` represents missing values.

**Convert to one-hot**:

Fields defined in the config YAML file, section `dataset`, sub-section `ont_hot` are converted into one-hot encoding.
Also, any categorical field that has a cardinality (i.e. number of categories) equal or less then `one_hot_max_cardinality` is converted to one-hot encoding.

If there are missing values, a column `*_isna` is added to the one-hot encoding.

**Missing data**:

In any column having missing values that was not converted to date, categorical or one-hot; a new column `*_na` is created (where the value is '1' if the field has missing a value) and the missing values are replaced by the median of the non-missing values.

**Drop low standard deviation fields**

All fields having standard deviation equal or lower than `std_threshold` (by default `0.0`) are dropped. Using the default value (`std_threshold=0.0`) this means dropping all fields having the exact same value for all samples.

### Dataset: Augment
This step invokes the user defined function `@augment` to perform dataset augmentation
1. If this step has already been performed (i.e. a dataset loaded from a pickle file that has already been augmented), the step is skipped
1. Invoke a user defined function decorated with `@dataset_augment`:
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, no error is produced.
    - Parameters: The first parameter is the dataset. Other parameters are defined in config YAML file section `functions`, sub-section `dataset_augment`
    - The return value replaces the original dataset
    - The dataset is marked to be saved

### Dataset: Preprocess
This step is used to pre-process data in order to make the dataset compatible with the inputs required by the model (e.g. normalize values)
1. If this step has already been performed (i.e. a dataset loaded from a pickle file that has already been pre-processed), the step is skipped
1. Invoke a user defined function decorated with `@dataset_preprocess`:
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, no error is produced.
    - Parameters: The first parameter is the dataset. Other parameters are defined in config YAML file section `functions`, sub-section `dataset_preprocess`
    - The return value replaces the original dataset
    - The dataset is marked to be saved

### Dataset: Split
This step us used to split the dataset into "train", "test" and "validation" datasets.
1. If this step has already been performed (i.e. a dataset loaded from a pickle file that has already been transformed), the step is skipped
1. Invoke a user defined function decorated with `@dataset_split`:
    1. If there is no user defined function `@dataset_split` or the section is disabled in the config file (i.e. `enable=False`), this step has failed (no error is produced) attempt to use a default implementation of "dataset split".
    - Parameters: The first parameter is the dataset. Other parameters are defined in config YAML file section `functions`, sub-section `dataset_split`
    - If the function is invoked, the return value must be a tuple of three datasets: `(dataset_train, dataset_test, dataset_validate)`
    - The return value from the function replaces the original dataset (specifically, each value replaces the train/test/validate datasets)
    - The dataset is marked to be saved
1. Attempt to use a default implementation of "dataset split"
    - If config YAML parameter `is_use_default_split` is set to `False`, the split step failed (no error is produced)
    - The default split implementation attempts to split the dataset in three parts, defined by config YAML file parameters `split_test` and `split_validate` in section `dataset_split`. If these parameters are not defined in the config YAML file, the split section failed (no error is produced)

### Dataset: Save
If the dataset is marked to be saved in any of the previous steps, attempt to save the dataset.

1. If the the YAML config variable `do_not_save` is set to `True`, this step is skipped
1. If a user defined function decorated with `@dataset_save` exists, it is invoked
    - Parameters: The first four parameters are `dataset, dataset_train, dataset_test, dataset_validate`. Other parameters are defined in config YAML file section `functions`, sub-section `dataset_save`
1. Otherwise the dataset is saved to the pickle file `{dataset_path}/{dataset_name}.pkl`

# Main workflow: Model

In these steps we create and train models. This also takes care of common tasks, such as hyper-parameter optimization, cross-validation and model analysis.

The main steps are:

1. Model Create: `@model_create`
1. Model Train: `@model_train`
1. Model Save: `@model_save`
1. Model Save train results
1. Model Test: `@model_evaluate`
1. Model Validate: `@model_evaluate`


A new `model_id` is created each time a new model is created/trained. This is used to make sure that files created during a run do not collision with other files names from previous runs. The `model_id` has the format `yyyymmdd_hhmmss.counter` where:
    - `yyyy`, `mm`, `dd`, `hh`, `mm`, `ss`: Current year, month, day, hour, minute, second (UTC time)
    - `counter`: Number of models created in this `Log(ML)` run (increasing counter starting with `1`).

**Logging**: All results from STDOUT and STDERR are saved to `{model_path}/{model_name}.parameters.{model_id}.stdout` and `{model_path}/{model_name}.parameters.{model_id}.stderr` respectively. Note that `model_id` is included in the path, so creating several models in the same `Log(ML)` run would save each output set to different `stdout/stderr` files (see details below).

###YAML config: Model section

```
model:
  model_name: 'MyModel'              # Model name: A simple string to use for file names related to this model
  model_path: 'path/to/dir'          # Train path: A path where to store logs and data from training
  is_save_model_pickle: False        # Try to save model using a pickle file?
  is_save_model_method: True         # Try to save model using a pickle file?
  is_save_model_method_ext: 'model'  # Model file extension
  is_save_test_pickle: True          # Save model test results to a pickle file?
  is_save_train_pickle: False        # Save model train results to pickle file?
  is_save_validate_pickle: False     # Save model validation results to pickle file?
```

### Model: Create

Create a new model, to be trained. It also saves the parameters used to create the model to a YAML file.

1. If a user defined function decorated with `@model_create` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameter is `dataset_train` if the dataset was split, otherwise is the full dataset. Other parameters are defined in config YAML file section `functions`, sub-section `model_create`
    - The return value from the user defined function is stored as the `model`
1. Current parameters are saved to a YAML file `{model_path}/{model_name}.parameters.{model_id}.yaml`. Note that `model_id` is included in the path, so creating several models in the same `Log(ML)` run would save each parameter set to different YAML files.

### Model: Train

Train the model.

1. If a user defined function decorated with `@model_train` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameters are `model` and `dataset_train` (if the dataset was split, otherwise is the full dataset). Other parameters are defined in config YAML file section `functions`, sub-section `model_train`
    - The return value from the user defined function is stored as the `train_results` (these result will be saved, see later steps)

### Model: Save

Save the (trained) model.

1. If a user defined function decorated with `@model_save` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program tries to save using a pickle file (see next step).
    - Parameters: The first parameters is the `model`. Other parameters are defined in config YAML file section `functions`, sub-section `model_save`
    - Return successful
1. Attempt to save model to pickle file if previous step (`@model_save` function) failed.
    - If parameter `is_save_model_pickle` from config YAML file is set to `False`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.model.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so training several models in the same `Log(ML)` run would save each model to different pickle files.
1. Attempt to save model to using `model.save()` if previous step failed.
    - If parameter `is_save_model_method` from config YAML file is set to `False`, this step is skipped
    - Invoke model's method `model.save({file_name})`, where `file_name` is set to `{model_path}/{model_name}.model.{model_id}.{is_save_model_method_ext}` (parameter `is_save_model_method_ext` is defined in config YAML file)

### Model: Save train Results

Save results from training to a pickle file

1. Attempt to save model training results (i.e. the return value from `@model_train` function) to pickle.
    - If parameter `is_save_train_pickle` from config YAML file is set to `False`, this step is skipped
    - If the results are `None`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.train_results.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so training several models in the same `Log(ML)` run would save train results to different pickle files.

### Model: Test

Evaluate the model on the `dataset_test` dataset_test

1. If a user defined function decorated with `@model_evaluate` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameters are `model` and `dataset_test` (if the dataset was split, otherwise use full dataset). Other parameters are defined in config YAML file section `functions`, sub-section `model_evaluate`
    - The return value from the user defined function is stored as the `test_results` (these result will be saved, see later steps)

### Model: Save test results

1. Attempt to save model test results (i.e. the return value from `@model_evaluate` function invoked with `dataset_test` parameter) to pickle.
    - If parameter `is_save_test_pickle` from config YAML file is set to `False`, this step is skipped
    - If the results are `None`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.test_results.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so testing several models in the same `Log(ML)` run would save train results to different pickle files.


### Model: Validate

Evaluate the model on the `dataset_validate` dataset_test

1. If a user defined function decorated with `@model_evaluate` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameters are `model` and `dataset_validate` (if the dataset was split, otherwise this step fails). Other parameters are defined in config YAML file section `functions`, sub-section `model_evaluate`
    - The return value from the user defined function is stored as the `validate_results` (these result will be saved, see later steps)

### Model: Save validate results

1. Attempt to save model test results (i.e. the return value from `@model_evaluate` function invoked with `dataset_validate` parameter) to pickle.
    - If parameter `is_save_validate_pickle` from config YAML file is set to `False`, this step is skipped
    - If the results are `None`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.validate_results.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so validating several models in the same `Log(ML)` run would save train results to different pickle files.


# Alternative workflows

There are some `Log(ML)` workflows that run "dataset" and "model" workflows several times:
1. Hyper-parameter optimization
1. Cross-validation

## Alternative workflow: Hyper-parameter optimization

This workflow allows to perform hyper-parameter optimization using a Bayesian framework (`hyper-opt`). The hyper parameters can be optimized in several stages of the "dataset" and "model".

The hyper-parameter optimization workflow adds a bayesian optimization on top of the main workflow. This means that, conceptually, it's executing the main workflow several times using a bayesian optimizer.

The hyper-parameter optimizaition method used is HyperOpt, for details see [Hyperopt documentation](http://hyperopt.github.io/hyperopt/)

Typically, hyper-parameter optimization is used to tune model training parameters. `Log(ML)` also allows to tune model creation parameters, as well as data augmentation and preprocessing parameters.

### YAML config
YAML configuration of hyper parameter optimization: All parameter are defined in the `hyper_parameter_optimization` section.

```
hyper_parameter_optimization:
    enable: False                   # Set this to 'True' or comment out to enable hyper-parameter optimization
    show_progressbar: True          # Show progress bar
    algorithm: 'tpe'                # Algorithm: 'tpe' (Bayesian Tree of Parzen Estimators), 'random' (random search)
    max_evals: 100                  # Max number of hyper-parameter evaluations. Keep in mnd that each evaluation is a full model training    # Parameter space to explore
    space:                          # Parameters search space specification, add one section for each user defined function you want to optimize
        dataset_augment:                    # Add parameter space specification for each part you want to optimize (see examples below)
        dataset_create:
        dataset_preprocess:
        model_create:
        model_train:
```

**Search space**: We define parameters for each part we want to optimize (e.g. `preprocess`, `model_create`, etc.).
The format for each parameter space is:
```
parameter_name: ['distribution', distribution)parameters...]
```
For distribution names and parameters, see: [section 'Parameter Expressions'](https://github.com/hyperopt/hyperopt/wiki/FMin)

Important: The parameters space definition should be a subset of the parameters in each `function` section.

Example: Perform hyper-parameter optimization of the learning rate using a uniform distribution as a p
```
hyper_parameter_optimization:
    algorithm: 'tpe'
    max_evals: 100
    space:
        model_train:
            learning_rate: ['uniform', 0.0, 0.5]
```

```
hyper_parameter_optimization:
    algorithm: 'tpe'
    max_evals: 100
    space:
        dataset_preprocess:
          num_x: ['choice', [100, 200, 300, 400, 500]]
          num_y: ['choice', [20, 50, 100, 200]]
        model_create:
            layer_1: ['randint', 10]
        model_train:
            learning_rate: ['uniform', 0.0, 0.5]
```

```
hyper_parameter_optimization:
    algorithm: 'tpe'
    max_evals: 100
    space:
        dataset_preprocess:
          num_x: ['choice', [100, 200, 300, 400, 500]]
          num_y: ['choice', [20, 50, 100, 200]]
        model_create:
            layer_1: ['randint', 20]
            layer_2: ['randint', 10]
        model_train:
            learning_rate: ['uniform', 0.0, 0.5]
```

## Alternative workflow: Cross-validation

This workflow is a Cross-Validation method built on top of the Train part of `Log(ML)` main workflow.


The YAML configuration is quite simple, you need to enable cross-validation and then specify the cross-validation type and the parameters:
The cross-validation workflow is implemented using SciKit learn's cross validation, on the methods and parameters see [SciKit's documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
```
cross_validation:
    enable: True	# Set this to 'True' to enable cross validation
    # Select one of the following algorithms and set the parameters
    KFold:
        n_splits: 5
    # RepeatedKFold:
    #     n_splits: 5
    #     n_repeats: 2
    # LeaveOneOut:
    # LeavePOut:
    #     p: 2
    # ShuffleSplit:
    #     n_splits: 5
    #     test_size: 0.25
```

# Alternative workflow: Data exploration

These steps implement feature exploration and importance analysis.

1. Feature statistics
1. Co-linearity analysis
1. Feature importance


# Command line argument

Command line options when invoking a `Log(ML)` program:

```
-c <config.yaml> : Specify a YAML config file
-d               : Debug mode, show lots of internal messages
-v               : Verbose
```


# Model Search
1. ada_boost_classifier
1. ada_boost_regressor
1. ard_regression
1. bagging_classifier
1. bagging_regressor
1. bayesian_ridge
1. bernoulli_nb
1. complement_nb
1. decision_tree_classifier
1. decision_tree_regressor
1. dummy_classifier_most_frequent
1. dummy_classifier_prior
1. dummy_classifier_stratified
1. dummy_classifier_uniform
1. dummy_regressor_mean
1. dummy_regressor_median
1. elastic_net_cv
1. extra_trees_classifier
1. extra_trees_regressor
1. gaussian_nb
1. gradient_boosting_classifier
1. gradient_boosting_regressor
1. hist_gradient_boosting_classifier
1. hist_gradient_boosting_regressor
1. huber_regressor
1. k_neighbors_classifier
1. k_neighbors_regressor
1. lars_regression
1. lasso_cv_regression
1. lasso_regression
1. linear_regression
1. linear_svc
1. linear_svr
1. logistic_regression_cv
1. multinomial_nb
1. nearest_centroid
1. nu_svc
1. nu_svr
1. orthogonal_matching_pursuit_regression
1. passive_aggressive_classifier
1. perceptron
1. radius_neighbors_classifier
1. radius_neighbors_regressor
1. random_forest_classifier
1. random_forest_regressor
1. ransac_regressor
1. ridge_cv_regression
1. ridge_regression
1. svc
1. svr
1. theil_sen_regressor


# Model Search with Hyper-parameter optimization:
1. extra_trees_classifier
1. extra_trees_regressor
1. gradient_boosting_classifier
1. gradient_boosting_regressor
1. k_neighbors_classifier
1. k_neighbors_regressor
1. random_forest_classifier
1. random_forest_regressor
