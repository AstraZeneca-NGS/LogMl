
# Datasets

The first steps in a `LogMl` workflow perform dataset operations:

1. Load: `@dataset_load`
1. Create (if not loaded): `@dataset_create`
1. Transform: `@dataset_transform`
1. Augment: `@dataset_augment`
1. Preprocess: `@dataset_preprocess`
1. Split: `@dataset_split`
1. Save: `@dataset_save`

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


## Dataset custom functions:

1. Load: `@dataset_load`
1. Create (if not loaded): `@dataset_create`
1. Transform: `@dataset_transform`
1. Augment: `@dataset_augment`
1. Preprocess: `@dataset_preprocess`
1. Split: `@dataset_split`
1. Save: `@dataset_save`
