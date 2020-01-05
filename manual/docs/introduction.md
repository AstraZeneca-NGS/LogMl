
# Introduction

To run LogMl you only need:

1. A dataset
1. A configuration (YAML) file

### Dataset

By far the easiest is to use a csv file.
Usually the file is loaded from `data/{dataset_name}/{dataset_name}.csv`, where `dataset_name` is some name (e.g. `my_dataset`).

### Configuration file

The configuration file is where you define basic information (e.g. the dataset name), which methods you want to run, algorithm parameters, etc.

For example, this is the configuration file for one of the introductory examples (file `config/intro.yaml`):

```
dataset:                                               # Section: Basic dataset information
  dataset_type: 'df'                                   # This dataset is a data frame
  dataset_name: 'intro'                                # Dataset name
  dataset_path: 'data/intro'                           # Path to dataset directory
  outputs: ['y']                                       # Name of the output variables

dataset_preprocess:                                    # Section: Preprocessing options
  impute:                                              # How to impute missing values
    median: true                                       # Impute using 'median'
  normalize:                                           # How to normalize inputs
    standard: true                                     # Use 'standard', i.e. transform so that mean=0 and std=1

model:                                                 # Section: Model definition
  model_type: regression                               # It's a regression model
  model_name: 'model_intro'                            # Model name
  model_class: sklearn.ensemble.RandomForestRegressor  # Use RandomForestRegressor from Scikit-learn
  model_path: 'data/intro/model'                       # Store models in this directory

functions:                                             # Section: User defined functions
    dataset_split:                                     # Split dataset parameters
      split_validate: 0.2                              #    Use 20% for validation set
      split_test: 0.0                                  #    No test set
    model_create:                                      # Model parameters: Same parameters as SkLearn RandomForestRegressor
      n_estimators: 30                                 #    Number of estimators
      min_samples_leaf: 3                              #    Minimum samples per leaf
      max_features: 0.7                                #    Max number of features
```

### Running LogMl

Once the dataset and configuration files are ready, you can simply run LogMl like this:

```
# Go to LogMl install dir, activate virtual environment
cd ~/logml
. ./bin/activate

# Run LogMl
# By default it will look for config file 'config.yaml' in the current directory
./src/logml.py
```
