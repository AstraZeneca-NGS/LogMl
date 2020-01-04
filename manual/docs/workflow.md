
# LogMl Workflow

LogMl workflow is shown in the diagram

![LogMl pipeline diagram](img/LogMl.png)

Note that some steps have a `name` to identify them, these steps can be customized with user_defined_functions

1. Dataset:
  	1. `dataset_load`: Load a dataset from a file
    1. `dataset_create`: Create a dataset using a user_defined_function
  	1. `dataset_preprocess`: Pre-process data, e.g. to make it suitable for model training.
  	1. `dataset_augment`: Data augmentation
  	1. `dataset_split`: Split data into train / validate / test datasets
  	1. `dataset_inout`: Obtain inputs and output for each (train / validate / test) dataset.
    1. `dataset_save`: Save datasets to a file
1. Explore: E.g. show dataset, correlation analysis, dendogram, missingness analysis, etc.
1. Feature importance:
1. Model:
    1. `model_create`: Create a new model
  	1. `model_train`: Train a model
  	1. Hyper-parameter optimization: Train models using hyper-parameter optimization using Bayesian or Random algorithms
  	1. Model Search: Search over 50 different model families (i.e. machine learning algorithms). This search can be done using hyper-parameter optimizations on each model's hyper-parameters.
    1. `model_evaluate`: Evaluate the model using train, validation or test datasets
    1. `model_predict`: Generate model predictions
    1. `model_save`: Save model to a file
1. Cross-validation: Perform cross-validation (KFold, RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit). Cross-validation can be used in all model search and Feature importance steps.
1. Logging: All outputs are directed to log files
1. Save models & datasets: Automatically save models and datasets for easier retrieval later

### Nomenclature

Note that these nomenclature items are separated by underscores instead of spaces, to denote a special meaning.

**LogMl_workflow**: The workflow the LogMl, specified in the previous section

**Workflow_step**: Each of steps in the **LogMl_workflow**

**Named_workflow_step**: A *workflow_step* that has a name (see previous section), e.g. The steps identified with `dataset_load`, `dataset_preprocess`, `model_train`, etc.

**workflow_step_name**: The name of a **Named_workflow_step**, e.g. `dataset_load`, `dataset_preprocess`, `model_train`, etc.

**User_defined_function**: Python code that has been annotated with a LogMl name (from a *named_workflow_step*).

**LogMl_default_function**: A function provided by LogMl, if no *user_defined_function* exists, the *LogMl_default_function* will be invoked instead

**Config_YAML**: A LogMl configuration YAML file. The YAML file can also store parameters that are passed to *user_defined_functions*

**Soft_fail**: When a *workflow_step* fails, warning might be shown, but *LogMl_workflow* execution continues

**Hard_fail**: When a *workflow_step* fails, an error message is shown and *LogMl_workflow* is stopped

**Processed_dataset_file**: LogMl saves datasets to (pickle) file after dataset steps (pre-processed, augmented, split), so that next time it ca be retrieved from the file instead of spending time on running the steps again.

# Customizing workflow steps

There are two ways to customize the workflow steps:

1. **User_defined_functions**: Custom (Python) code. This can be done on *named_workflow_steps*, such as `dataset_load`, `dataset_preprocess`, `model_train`, etc.
1. **Config_YAML**: Configuration parameters are stored in a YAML file.

How each LogMl *named_workflow_step* works:

1. The user can provide a *user_defined_function* (custom Python code) for each *named_workflow_steps*. E.g. For the `dataset_load` workflow step, a user provides a function to load the dataset
1. If there is no user_defined_function, a *LogMl_default_function* is used. E.g. For the `dataset_load` workflow step, LogMl tries to load a dataset from a CSV file
1. If there is neither a *user_defined_function*, nor a *LogMl_default_function*, the step fails (can be *soft_fail* or *hard_fail*, depending on the step)

### User_defined_function

A *user_defined_function* is just Python code that has been annotated with a LogMl name (from a *named_workflow_step*).
The *user_defined_functions* are often referred by the names with an `@` prefix, for instance `@dataset_load` refers to the *user_defined_function* annotated with `dataset_load` *workflow_step_name*

Sample code using *user_defined_function* to create a dataset:

```
#!/usr/bin/env python3

import numpy as np
from logml import *

# This is a user_defined_function (it is annotated with a LogMl 'named_workflow_step')
@dataset_create
def my_dataset_create(num_samples):
	x = 2 * np.random.rand(num_samples, 2) - 1
	y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype('float').reshape(num_samples, 1)
	return np.concatenate((x, y), axis=1)

# Main
lm = LogMl()  # Create LogMl object
lm()          # Invoke it to execute the LogMl workflow (LogMl objects are callable)
```

### Config_YAML

The *user_defined_function* we saw in the previous section (`my_dataset_create`) has a parameter `num_samples`.
We can define the parameter in the *config_YAML* file:

```
dataset:
  dataset_name: 'example_01'
  dataset_path: 'data/example_01'

model:
  enable: True
  model_name: 'example_01'
  model_path: 'data/example_01/model'

functions:
    dataset_create:           # Parameters in this sub-section are passed
      num_samples: 1000       # to @dataset_create user_defined_function
```

Note that the `functions` section in the *config_YAML* file, has a `dataset_create` sub-section.
All parameters defined in that sub-section are passed as named parameters to the Python `my_dataset_create` function (i.e the *user_defined_function* annotated with `dataset_create`).

Given the code and YAML shown above, when LogMl executes the wokflow and reaches the `dataset_create` *workflow_step*, it will invoke `my_dataset_create(1000)` and store the return value as the dataset.
The dataset will then be used in all the next *workflow_steps*: it will be pre-processed, split into train/validation/test, used for model training, etc.

**LogMl_default_functions**: Creating custom functions is very flexible, but many times LogMl already has good default functions you can use.
Particularly, when using a DataFrame (e.g. a dataset provided as CSV file), there are many functionalities that LogMl provides by default.
These *LogMl_default_functions* have many parameters that are defined in the *config_YAML*
