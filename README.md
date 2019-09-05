
# `Log(ML)`

Log(ML) is a framework that helps automate some steps in machine learning projects.

**Why?**
There is a considerable amount is setup, boiler-plate code, analysis in every ML project.
`Log(ML)` performs most of these boring tasks, so you can focus on what's important and adds value.

**What does Log(ML) do for me?**
Log(ML) can:
- Facilitates logging in ML projects: No more writing down results in a notepad, `Log(ML)` creates log file in a systematic manner
- Save models and results: `Log(ML)` saves all your models, so you can always retrieve the best ones.
- Splitting datasets:

**How does Log(ML) work?**
`Log(ML)` has a standard "data processing" workflows. These workflows include several steps, such as data preprocessing, data augmentation, data exploration, model training, hyper-parameter search, cross-validation, etc. Each step in the workflow can be customized.

# Nomenclature

Parameters from YAML: We refer to parameters defined in YAML file as between curly brackets, e.g. `{parameter_name}`

User defined functions: This are functions defined by the user and marked with the `Log(ML)` annotations. For instance, the "user function decorated with `@dataset_load`" is sometimes referred as the "`@dataset_load` function", for short


# Workflows

`Log(ML)` performs the following series of steps (all of them customizable using Python functions and YAML configuration). `Log(ML)` allows you to define your own custom functions by adding annotations.

The "Main workflow" is the default workflow that `Log(ML)` executes and builds on to provide additional functionality (provided in the "Alternative workflows")

Here is a summary of the "Main workflow" steps (details are covered in the next sub-sections):

1. Dataset: load or create dataset, transform, augment, preprocess, split
1. Training: create model, train, hyper-parameter optimization, cross-validation, etc.

Alternative workflows: These workflows provide additional functionality on top of the "Main workflow"
1. Hyper parameter optimization
1. Cross validation
1. Data exploration
1. Model analysis

# Learning by examples

This is Machine Learning, so let's learn by showing some examples...(hopefully you can generalize as well as your ML algorithms)

In this section we introduce some examples on how to use `Log(ML)` and show how the framework simplifies some aspect fo machine learning projects.

### Basic setup

`Log(ML)` can provide some default implementations for some steps of the workflow, but others you need to provide yourself (e.g. code to create your machine learning model). These steps are provided in the Python code you write.

Both your Python code and the default `Log(ML)` implementations require parameters, these parameters are configured in a YAML file.

So, a `Log(ML)` project consist of (at least) two parts:
1. A Python program
1. A YAML configuration file

### Example 1: A neural network for "XOR"

In the code shown in `example_01.py` (see below)
we train a neural network model to learn the "XOR" problem. We create three functions:
- `my_dataset_create`: Create a dataset (a NumPy matrix) having the inputs and outputs for our problem. We create two columns (the inputs) of `num_samples` row or random numbers in the interval `[-1, 1]`. The third column (the output) is the "XOR" of the first two columns
- `my_model_create`: Create a neural network using Tenforflow and Keras sequential mode. The network one hidden layer with `num_neurons` neurons
- `my_model_train`: Train the neural network using a learning rate of `learning_rate` and `epochs` number of epochs.
- `my_model_eval`: Evaluate the neural network.

Note that the functions are decorated using `Log(ML)` decorators `@dataset_create`, `@@model_create`, `@model_train` , `@model_evaluate`

Python code `example_01.py`:
```
#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from logml import *
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad

@dataset_create
def my_dataset_create(num_samples):
	x = 2 * np.random.rand(num_samples, 2) - 1
	y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype('float').reshape(num_samples, 1)
	return np.concatenate((x, y), axis=1)

@model_create
def my_model_create(dataset, num_neurons):
	model = Sequential()
	model.add(Dense(num_neurons, activation='tanh', input_shape=(2,)))
	model.add(Dense(1, activation='tanh'))
	return model

@model_train
def my_model_train(model, dataset, learning_rate, epochs):
	model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error')
	return model.fit(dataset[:, 0:2], dataset[:, 2], epochs=epochs)

@model_evaluate
def my_model_eval(model, dataset):
	return model.evaluate(dataset[:, 0:2], dataset[:, 2])

ml = LogMl()
ml()
```

We also need to create a configuration YAML file (see below). This YAML file defines three sections:
- `dataset`: Defines the name of the dataset and path to save dataset files.
- `train`: Defines the name of the model and path to save model, model parameters and training results files.
- `functions`: These define the values to pass to the functions defined in our python program (or `Log(ML)` default implementations).

Configuration YAML file `example_01.yaml`
```
dataset:
  dataset_name: 'example_01'
  dataset_path: 'data/example_01'

model:
  model_name: 'example_01'
  model_path: 'data/example_01/model'

functions:
  dataset_create:
    num_samples: 1000
  dataset_split:
    split_test: 0.2
    split_validate: 0.0
  model_create:
      num_neurons: 3
  model_train:
    epochs: 20
    learning_rate: 0.3
```
A few remarks about the `functions` section:
1. The name of the parameters in the YAML must match exactly the name of the respective Python functions parameters
1. Python annotation matches the subsection in the YAML file (e.g. parameters defined YAML subsection `dataset_create` is called `num_samples`, which matches the parameter of the Python function annotated with `@dataset_create`)
1. Since our `@model_evaluate` function doesn't take any additional arguments than the ones provided by `Log(ML)` (i.e. `model` and `dataset`), we don't need to specify the sub-sections in our YAML file
1. The `@dataset_split` function was not implemented in our program, so `Log(ML)` will provide a default implementation. This default implementation uses the parameters `split_test` and `split_validate` (the dataset is split according to these numbers)


Now we can run the program:
```
# By default the expected config file name is "ml.yaml" so we provide an alternative name name with command line option "-c"

$ ./example_01.py -c example_01.yaml
Epoch 1/20
1000/1000 [==============================] - 0s 178us/sample - loss: 0.2416
Epoch 2/20
1000/1000 [==============================] - 0s 30us/sample - loss: 0.1588
...
Epoch 20/20
1000/1000 [==============================] - 0s 30us/sample - loss: 0.0949
```

So, `Log(ML)` performed a workflow that:
1. Invoked the function to create a dataset using the arguments from the YAML file (i.e. `my_dataset_create(num_samples=20)`)
1. Invoked the function to create a model using as arguments the `dataset` plus the parameters from the YAML file (i.e. `my_model_create(dataset, num_neurons=3)`)
1. Invoked the function to train the model using as arguments the `model`, the `dataset` plus the parameters from the YAML file (i.e. `my_model_train(model, dataset, learning_rate=0.3, epochs=20)`)
1. Invoked the function to validate the model (evaluate on the validation dataset split) using only as arguments `model`, and `dataset_validate` (since there are no additional parameters from the YAML file)

But `Log(ML)` it also did log a lot of information that is useful for future references. In this case, it saved the dataset to a pickle file (`example_01.pkl`), the all parameters used to create and train this model (`data/example_01/train/example_01.parameters.20190823.212609.830649.1.yaml`) and the full STDOUT/STDERR (`data/example_01/train/example_01.20190823.212609.830649.1.stdout` and `data/example_01/train/example_01.20190823.212609.830649.1.stderr`)
```
$ ls data/example_01/* data/example_01/train/*
data/example_01/example_01.pkl
data/example_01/train/example_01.20190823.212609.830649.1.stdout
data/example_01/train/example_01.20190823.212609.830649.1.stderr
data/example_01/train/example_01.parameters.20190823.212609.830649.1.yaml
```

Now we can change the parameters in the YAML file (for instance set `learning_rate: 0.1`) and run the program again.
```
$ ./example_01.py -c example_01.yaml
Epoch 1/20
1000/1000 [==============================] - 0s 184us/sample - loss: 0.2561
...
Epoch 20/20
1000/1000 [==============================] - 0s 23us/sample - loss: 0.1112
```

All the new log files will be created and we can keep track of our project and the parameters we used.
OK, this model is not as good as the previous one, but fortunately we have all the logging information, so we don't have to remember the parameters we used for the best model.
```
$ ls data/example_01/* data/example_01/train/*
data/example_01/example_01.pkl
data/example_01/train/example_01.20190823.213803.075040.1.stdout
data/example_01/train/example_01.20190823.212609.830649.1.stderr
data/example_01/train/example_01.parameters.20190823.212609.830649.1.yaml
data/example_01/train/example_01.20190823.212609.830649.1.stdout
data/example_01/train/example_01.parameters.20190823.213803.075040.1.yaml
data/example_01/train/example_01.20190823.213803.075040.1.stderr
```
### Example 2: Hyper-parameter optimization

Building on the previous example (`example_01.py` and `example_01.yaml`), let's assume that instead of trying to tune the `learning_rate` manually, we'd prefer to perform hyper-parameter optimization.

In this example (`example_02`), we'll set up hyper-parameter optimization on `learning_rate`. The python program remains exactly the same as in the previous example, we'll be adding a hyper-parameter optimization section to the YAML file.

For the config YAML file (see `example_02.yaml`), we jut add the following section:
```
hyper_parameter_optimization:
    algorithm: 'tpe'
    max_evals: 100
    space:
        model_train:
          learning_rate: ['uniform', 0.0, 0.5]
```
We added a `hyper_parameter_optimization` section where we:
- Define the hyper parameter algorithm (`tpe`) which is a Bayesian apprach
- Set the number of evaluations to `100`
- Define that we want to optimize the parameter `learning_rate` in the function `@model_train` using a uniform prior in the interval `[0.0, 0.5]`.

We run the program:
```
$ ./example_02.py -c example_02.yaml

100%|██████████| 10/10 [00:06<00:00,  1.44it/s, best loss: 0.07341234689950943]
```

Here the hyper-parameter optimization is saying that the best loss found (with ten iterations) is `0.0734`.

We also have all the parameter details, models, and STDOUT/STDERR for every single model created and trained:
```
$ ls data/example_02/* data/example_02/train/* | cat
data/example_02/example_02.pkl
data/example_02/train/example_02.20190823.215947.132156.1.stderr
data/example_02/train/example_02.20190823.215947.132156.1.stdout
...
data/example_02/train/example_02.20190823.215953.151580.10.stderr
data/example_02/train/example_02.20190823.215953.151580.10.stdout
data/example_02/train/example_02.hyper_param_search.20190823.215953.151580.10.pkl
data/example_02/train/example_02.parameters.20190823.215947.132156.1.yaml
...
data/example_02/train/example_02.parameters.20190823.215953.151580.10.yaml
```

### Example 3: Neural network architecture optimization

Now we build on the previous example (Example 2) by trying to optimize the neural network architecture. For this we just need to add a hyper parameter optimization when building the neural network (i.e. the `@model_create` step in the workflow). Simply add a line in the `space` definition within `hyper_parameter_optimization` section:

The YAML is changed like this (see `example_03.yaml`):
```
hyper_parameter_optimization:
    ...
    space:
        model_create:
          num_neurons: ['randint', 5]
        ...
```

Also we need a minor change in the python program is to ensure that we at least have one neuron in the hidden layer (otherwise the model doesn't make sense) So we add a single line to `@model_create` (see line `num_neurons = max(num_neurons, 1)` below):
```
@model_create
def my_model_create(dataset, num_neurons):
	model = Sequential()
	num_neurons = max(num_neurons, 1)                                  # <-- Added this line
	model.add(Dense(num_neurons, activation='tanh', input_shape=(2,)))
	model.add(Dense(1, activation='tanh'))
	return model
```

That's is, we have network architecture optimization (`num_neurons`) and hyper-parameter optimization (`learning_rate`). Let's run the program (output edited for readability):

```
$ ./example_03.py -v -c example_03.yaml
...
2019-08-23 21:29:51,924 INFO Hyper parameter optimization:	iteration: 10	...
    best fit: 0.06886020198464393
    best parameters: {'model_create': {'num_neurons': 3}, 'model_train': {'learning_rate': 0.22890998206259194}}
...
```
The best parameters, for a 10 iteration hyper-optimization, are `num_neurons=3` and `learning_rate=0.2289`.

# Main workflow: Overview

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
