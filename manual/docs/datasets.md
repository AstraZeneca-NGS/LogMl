
# Datasets

The first steps in a *LogMl_workflow* perform dataset operations.
There are several *named_workflow_steps* taking care of different aspects of dataset processing to make sure is suitable for training a model

1. `dataset_load`: Load a dataset from a file
1. `dataset_create`: Create a dataset using a user_defined_function
1. `dataset_preprocess`: Pre-process data, e.g. to make it suitable for model training.
1. `dataset_augment`: Data augmentation
1. `dataset_split`: Split data into train / validate / test datasets
1. `dataset_inout`: Obtain inputs and output for each (train / validate / test) dataset.
1. `dataset_save`: Save datasets to a file


### Config_YAML

There are several parameters that can be defined in the *config_YAML*:

```
dataset:
  # Dataset type: 'df' means dataFrame
  dataset_type: 'df'

  # Dataset name:
  #   A simple name for the dataset.
  #   It is appended to dataset_path to create dataset_file_name
  dataset_name: 'my_dataset'

  # Dataset Path: Path to use when loading and saving datasets
  dataset_path: 'data/my_dataset'

  # If set, loading dataset from pickle file will be skipped
  do_not_load_pickle: false

  # If set, saving dataset will be skipped
  do_not_save: false

  # Use all inputs, no outputs.
  # I.e. do not split in/out.
  # E.g. unsupervised learning
  is_use_all_inputs: false

  # Use default 'in_out' method
  is_use_default_in_out: true

  # Use default preprocess
  is_use_default_preprocess: true

  # Use (internal) 'split' function if none is provided by the user?
  # Note: This function returns dataset as a list
  is_use_default_split: false

  # Use default transform operation
  is_use_default_transform: true

  # Output variables, a.k.a. dependent variables (i.e. what we want to predict)
  outputs: []
```

### Dataset: Load
This step typical attempts to load data from files (e.g. load a "data frame" or a set of images).

1. Attempt to load from pickle file (`{dataset_path}/{dataset_name}.pkl`). If the files exists, load the dataset.
1. Invoke a *user_defined_function* decorated with `@dataset_load`.
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, so LogMl will attempt to create a dataset (next step)
    - Parameters to the *user_defined_function* are defined in *config_YAML* file section `functions`, sub-section `dataset_load`
    - The dataset is marked to be saved

### Dataset: Create
This part creates a dataset by invoking the *user_defined_function* decorated by `@dataset_create`

1. If the dataset has been loaded (i.e. "Load dataset" step was successful), skip this step
1. Invoke a *user_defined_function* decorated with `@dataset_create`:
    - If there is no *user_defined_function* or the section is disabled in the config file (i.e. `enable=False`), this step *hard_fails*, i.e. since LogMl doesn't have a dataset to work with (load and create steps both failed) it will exit with an error.
    - Parameters to the *user_defined_function* are defined in *config_YAML* file section `functions`, sub-section `dataset_create`
    - The return value from the *user_defined_function* is used as a dataset
    - The dataset is marked to be saved

### Dataset: Preprocess
This step is used to pre-process data in order to make the dataset compatible with the inputs required by the model (e.g. normalize values)

1. If this step has already been performed (i.e. a dataset loaded from a pickle file that has already been pre-processed), the step is skipped
1. Invoke a *user_defined_function* decorated with `@dataset_preprocess`:
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, no error is produced.
    - Parameters: The first parameter is the dataset. Other parameters are defined in *config_YAML* file section `functions`, sub-section `dataset_preprocess`
    - The return value replaces the original dataset
    - The dataset is marked to be saved


### Dataset: Augment
This step invokes the *user_defined_function* `@augment` to perform dataset augmentation

1. If this step has already been performed (i.e. a dataset loaded from a pickle file that has already been augmented), the step is skipped
1. Invoke a *user_defined_function* decorated with `@dataset_augment`:
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, no error is produced.
    - Parameters: The first parameter is the dataset. Other parameters are defined in *config_YAML* file section `functions`, sub-section `dataset_augment`
    - The return value replaces the original dataset
    - The dataset is marked to be saved

### Dataset: Split
This step us used to split the dataset into "train", "test" and "validation" datasets.

1. If this step has already been performed (i.e. a dataset loaded from a pickle file that has already been split), the step is skipped
1. Invoke a *user_defined_function* decorated with `@dataset_split`:
    1. If there is no *user_defined_function* `@dataset_split` or the section is disabled in the config file (i.e. `enable=False`), this step has failed (no error is produced) attempt to use a *LogMl_default_function* of `dataset_split`.
    - Parameters: The first parameter is the dataset. Other parameters are defined in *config_YAML* file section `functions`, sub-section `dataset_split`
    - If the function is invoked, the return value must be a tuple of three datasets: `(dataset_train, dataset_test, dataset_validate)`
    - The return value from the function replaces the original dataset (specifically, each value replaces the train/test/validate datasets)
    - The dataset is marked to be saved
1. Attempt to use a *LogMl_default_function* of `dataset_split`
    - If *config_YAML* parameter `is_use_default_split` is set to `False`, the split step failed (no error is produced)
    - The default split implementation attempts to split the dataset in three parts, defined by *config_YAML* file parameters `split_test` and `split_validate` in section `dataset_split`. If these parameters are not defined in the *config_YAML* file, the split section failed (no error is produced)

### Dataset: Inputs / Outputs

1. For each dataset splie (train, validation, and test):
    1. Invoke a *user_defined_function* decorated with `@dataset_inout`:
        1. If there is no *user_defined_function* `@dataset_inout` or the section is disabled in the config file (i.e. `enable=False`), this step fails (no error is produced) and LogMl attempts to use a *LogMl_default_function* for `dataset_inout`.
        - Parameters: The first parameter is the dataset, the second parameter is the datset name (`train`, `validate`, or `test`). Other parameters are defined in *config_YAML* file section `functions`, sub-section `dataset_inout`
        - If the function is invoked, the return value must be a tuple: `(inputs, outputs)`
        - The return value from the function replaces the original dataset (specifically, each value replaces the train/test/validate datasets)
        - The dataset is marked to be saved
    1. Attempt to use a *LogMl_default_function* of `dataset_inout`
        - If *config_YAML* parameter `is_use_default_in_out` is set to `False`, the split step *hard_fails* (execution is halted)
        - The default implementation only returns the same dataset is the *config_YAML* parameter `is_use_all_inputs` is set. Otherwise it *hard_fails*.

### Dataset: Save
If the dataset is marked to be saved in any of the previous steps, attempt to save the dataset.

1. If the the YAML config variable `do_not_save` is set to `True`, this step is skipped
1. If a *user_defined_function* decorated with `@dataset_save` exists, it is invoked
    - Parameters: The first four parameters are `dataset, dataset_train, dataset_test, dataset_validate`. Other parameters are defined in *config_YAML* file section `functions`, sub-section `dataset_save`
1. Otherwise the dataset is saved to the pickle file `{dataset_path}/{dataset_name}.pkl`

# DataFrames

There are several useful *LogMl_default_function* then the dataset is a DataFrame, i.e. when *config_YAML* defines `dataset_type='df'`.


### Dataset (df): Default Load

Load DataFrame from a CSV file using (using `pandas.read_csv`).

### Dataset (df): Default Preprocess

The default method for DataFrame pre-processing does:

1. Sanitize variables names: Convert names so that characters outside the set `[_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]` are converted to `_` (underscore)
1. Convert to categorical: Fields defined in the *config_YAML* file, section `dataset`, sub-section `categories` are converted into categorical data and converted to a numerical (integer) representation. Category `0` represents missing values.
1. Convert to one-hot: Fields defined in the *config_YAML* file, section `dataset`, sub-section `ont_hot` are converted into one-hot encoding. Also, any categorical field that has a cardinality (i.e. number of categories) equal or less then `one_hot_max_cardinality` is converted to one-hot encoding. If there are missing values, a column `*_isna` is added to the one-hot encoding.
1. Missing data indicators: In any column having missing values that was not converted to date, categorical or one-hot; a new column `*_na` is created (where the value is '1' if the field has missing a value) and the missing values are replaced by the median of the non-missing values.
1. Expand date/time features: Fields defined in the *config_YAML* file, section `dataset`, sub-section `dates` are treated as date/time when the dataFrame CSV is loaded and then expanded into several columns: `[Year, Month, Day, DayOfWeek, Hour, Minute, Second, etc.]`.
1. Remove samples with missing outputs: Can be disables using *config_YAML* option `remove_missing_outputs`
1. Drop low standard deviation fields: All fields having standard deviation equal or lower than `std_threshold` (by default `0.0`) are dropped. Using the default value (`std_threshold=0.0`) this means dropping all fields having the exact same value for all samples.
1. Remove duplicated inputs (e.g. two columns that are exactly the same)
1. Shuffle samples: Can be enabled using *config_YAML* option `shuffle`
1. Balance dataset: re-sample categories having low number of samples. Can be enabled using *config_YAML* option `balance`
1. Impute missing values. Can use several strategies: `[mean, median, most_frequent, one, zero]`
1. Normalize variables: Can use several strategies: `[standard, maxabs, minmax, minmax_neg, log, log1p, log_standard, log1p_standard, quantile]`

**Config_YAML Options** for datasets (df) pre-process

```
dataset_preprocess:
  # Set to 'false' to disable this step
  enable: true

  # Categorical data
  # A list of categorical input variables
  # Can be followed by a list of categories to enforce some type of categorical order in the codes
  categories:
    UsageBand: ['Low', 'Medium', 'High']
    ProductSize: ['Mini', 'Small', 'Compact', 'Medium', 'Large / Medium', 'Large']

  # Categorical data: Match field names using regular expressions
  # A list of categorical input variables
  # Can be followed by a list of categories to enforce some type of categorical order in the codes
  categories_regex:
    - 'zzz_.*': ['low', 'mid', 'high']
    - 'yyy_.*': True

  # List of data columns that are transformed into 'date/time'
  # These columns are also split into 'date parts' (year, month, day, day_of_week, etc.)
  dates: ['saledate']

  # Sanitize column names: Convert column names so that characters outside the set [_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789] are converted to '_' (underscore)
  is_sanitize_column_names: True

  # One hot encoding: List of variable to transform to one_hot
  ont_hot: ['Enclosure_Type']

  # One hot encoding: Encode all variables having a cardinality less or equal to 'one_hot_max_cardinality'
  one_hot_max_cardinality: 7

  # Remove repeated inputs (i.e. input variables that have the exact same values as other inputs)
  remove_equal_inputs: true

  # Remove rows having missing output/s
  remove_missing_outputs: true

  # Remove columns: List of columns to remove from dataFrame
  remove_columns: []

  # Remove columns: List of columns to remove from dataFrame (after all transformations have been applied)
  remove_columns_after: []

  # Shuffle samples (e.g. shuffle rows in dataframe)
  shuffle: false

  # Drop inputs having standard deviation below this limit
  std_threshold: 0.0001

  # Balance an unbalanced datasets (classification models only, since outputs must be cathegorical)
  balance: false

  # Impute values
  # Imputation method name, followed by list of variables, regex or 'true' (this last option means "use as default method")
  # that all variables should be imputed using this method
  impute:
    # Use the mean value
    mean:  ['x2', 'x5', 'xx.*']

    # Impute using the median value
    median: true

    # Impute using the most frequent value
    most_frequent: ['x3', 'x4']

    # Impute by assigning value '1'
    one: ['x9', 'o.*']

    # Do not impute these variables
    skip: ['x7', 'x8']

    # Impute by assigning value '0'
    zero: ['x10', 'z.*']


  # Normalize values
  # Normalization methods followed by list of variables, regex or 'true' meaning
  # that all variables should be normalized using this method
  normalize:
    # Transform so that mean(x)=0 var(x)=1
    # Set to 'true' means to use as default (i.e. use this normalization
    # for all variables not explicitly define elsewhere)
    standard: true

    # Normalize dividing by max(abs(x_i))
    maxabs: ['x2', 'x5', 'xx.*']

    # Use min/max normalizer, i.e. transform to interval [0,1]
    minmax: ['x3']

    # Use min/max normalizer with negative values, i.e. transform to interval [-1,1]
    minmax_neg: ['x6', 'neg_.*']

    # Apply log(x)
    log: ['x9']

    # Apply log(x+1)
    log1p: ['x10', 'x11']

    # Apply log(x), then use standard transform: (log(x) - mean(log(x))) / std(log(x))
    log_standard: ['x9']

    # Apply log(x+1), then use standard transform: (log(x+1) - mean(log(x+1))) / std(log(x+1))
    log1p_standard: ['x10', 'x11']

    # Quantile transformation: This method transforms the features to follow a uniform or a normal distribution
    quantile: ['q.*']

    # Do not normalize these variables
    skip: ['x7', 'x8']
```

### Dataset (df): Default Augment

The *LogMl_default_function* for dataset augmentation focusses on adding more columns to the dataset (i.e. augment the input variables), as opposed to augmenting the number of samples.

1. Add Principal components (PCA values): Can be customized to act on several groups of variables.
1. Add Non-negative Matrix Factorization components (NMF values): Can be customized to act on several groups of variables.

**Config_YAML Options** for datasets (df) augment:
```
dataset_augment:
  # Set to 'false' to disable this step
  enable: true

  # Add principal componenets
  # After the 'pca', you can add several "sub-sections":
  #   name:                 # Unique name, used to add PCA fields on each sample (name_1, name_2, ... name_num)
  #     num: 2              # Number of PCA componenets to add
  #     fields: ['x.*']     # List of regular expression to match field names (only fields matching this list will be used)
  pca:
    pca_dna:
      num: 2
      fields: ['DNA_.*']
    pca_log2:
      num: 3
      fields: ['log2ratio_.*', '.*_LOG2F']
  # Add NMF (non-negative matrix factorization)
  # After the 'nmf', you can add several "sub-sections":
  #   name:                 # Unique name, used to add NMF fields on each sample (name_1, name_2, ... name_num)
  #     num: 2              # Number of NMF componenets to add
  #     fields: ['x.*']     # List of regular expression to match field names (only fields matching this list will be used)
  nmf:
    nmf_dna:
      num: 2
      fields: ['DNA_.*']
    nmf_expr:
      num: 3
      fields: ['expr_.*', '.*_EXPR']
```
