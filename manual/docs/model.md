
# Workflow: Model

In these steps we create and train models. This also takes care of common tasks, such as hyper-parameter optimization, cross-validation and model analysis.

The main steps are:

1. `model_create`: Model Create
1. `model_train`: Model Train
1. `model_save`: Model Save
1. Model Save train results
1. `model_evaluate`: Model Test
1. `model_evaluate`: Model Validate

A new `model_id` is created each time a new model is created/trained. This is used to make sure that files created during a run do not collision with other files names from previous runs. The `model_id` has the format `yyyymmdd_hhmmss.counter` where:
    - `yyyy`, `mm`, `dd`, `hh`, `mm`, `ss`: Current year, month, day, hour, minute, second (UTC time)
    - `counter`: Number of models created in this `Log(ML)` run (increasing counter starting with `1`).

**Logging**: All results from STDOUT and STDERR are saved to `{model_path}/{model_name}.parameters.{model_id}.stdout` and `{model_path}/{model_name}.parameters.{model_id}.stderr` respectively. Note that `model_id` is included in the path, so creating several models in the same `Log(ML)` run would save each output set to different `stdout/stderr` files (see details below).

### Model: Create

Create a new model, to be trained. It also saves the parameters used to create the model to a YAML file.

1. If a user defined function decorated with `@model_create` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameter is `dataset_train` if the dataset was split, otherwise is the full dataset. Other parameters are defined in *config_YAML* file section `functions`, sub-section `model_create`
    - The return value from the user defined function is stored as the `model`
1. Current parameters are saved to a YAML file `{model_path}/{model_name}.parameters.{model_id}.yaml`. Note that `model_id` is included in the path, so creating several models in the same `Log(ML)` run would save each parameter set to different YAML files.

### Model: Train

Train the model.

1. If a user defined function decorated with `@model_train` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameters are `model` and `dataset_train` (if the dataset was split, otherwise is the full dataset). Other parameters are defined in *config_YAML* file section `functions`, sub-section `model_train`
    - The return value from the user defined function is stored as the `train_results` (these result will be saved, see later steps)

### Model: Save

Save the (trained) model.

1. If a user defined function decorated with `@model_save` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program tries to save using a pickle file (see next step).
    - Parameters: The first parameters is the `model`. Other parameters are defined in *config_YAML* file section `functions`, sub-section `model_save`
    - Return successful
1. Attempt to save model to pickle file if previous step (`@model_save` function) failed.
    - If parameter `is_save_model_pickle` from *config_YAML* file is set to `False`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.model.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so training several models in the same `Log(ML)` run would save each model to different pickle files.
1. Attempt to save model to using `model.save()` if previous step failed.
    - If parameter `is_save_model_method` from *config_YAML* file is set to `False`, this step is skipped
    - Invoke model's method `model.save({file_name})`, where `file_name` is set to `{model_path}/{model_name}.model.{model_id}.{is_save_model_method_ext}` (parameter `is_save_model_method_ext` is defined in *config_YAML* file)

### Model: Save train Results

Save results from training to a pickle file

1. Attempt to save model training results (i.e. the return value from `@model_train` function) to pickle.
    - If parameter `is_save_train_pickle` from *config_YAML* file is set to `False`, this step is skipped
    - If the results are `None`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.train_results.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so training several models in the same `Log(ML)` run would save train results to different pickle files.

### Model: Test

Evaluate the model on the `dataset_test` dataset_test

1. If a user defined function decorated with `@model_evaluate` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameters are `model` and `dataset_test` (if the dataset was split, otherwise use full dataset). Other parameters are defined in *config_YAML* file section `functions`, sub-section `model_evaluate`
    - The return value from the user defined function is stored as the `test_results` (these result will be saved, see later steps)

### Model: Save test results

1. Attempt to save model test results (i.e. the return value from `@model_evaluate` function invoked with `dataset_test` parameter) to pickle.
    - If parameter `is_save_test_pickle` from *config_YAML* file is set to `False`, this step is skipped
    - If the results are `None`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.test_results.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so testing several models in the same `Log(ML)` run would save train results to different pickle files.


### Model: Validate

Evaluate the model on the `dataset_validate` dataset_test

1. If a user defined function decorated with `@model_evaluate` exists, it is invoked
    - If there is no function or the section is disabled in the config file (i.e. `enable=False`), this step has failed, the program exits with an error.
    - Parameters: The first parameters are `model` and `dataset_validate` (if the dataset was split, otherwise this step fails). Other parameters are defined in *config_YAML* file section `functions`, sub-section `model_evaluate`
    - The return value from the user defined function is stored as the `validate_results` (these result will be saved, see later steps)

### Model: Save validate results

1. Attempt to save model test results (i.e. the return value from `@model_evaluate` function invoked with `dataset_validate` parameter) to pickle.
    - If parameter `is_save_validate_pickle` from *config_YAML* file is set to `False`, this step is skipped
    - If the results are `None`, this step is skipped
    - The model resulting from training is saved to a pickle file file `{model_path}/{model_name}.validate_results.{model_id}.pkl`.
    - Note that `model_id` is included in the path, so validating several models in the same `Log(ML)` run would save train results to different pickle files.
