
# Overview

`Log(ML)` performs the following series of steps (all of them customizable using Python functions and YAML configuration).

Here is an overview of the workflow steps (details are covered in the next sub-sections):

1. Dataset: Load or Create, Transform, Preprocess, Augment, Explore, Split, Inputs/Outputs
1. Feature importance
1. Model Training
	1. Hyper-parameter optimization
1. Model Search
1. Cross-validation

Each section can be enabled / disabled and customized in the YAML configuration file.
`Log(ML)` also allows you to define your own custom Python functions for each step.

# Basics

Usually the datasets are provided to `LogMl` in a form of a DataFrame, for instance, let's use the Iris dataset:

```
# Example for Iris
```

# Jupyter Notebooks

Here is an example of using `LogMl` from a Jupyter Notebook:

```
```

Output can be seen

!!!!!!HERE!!!!!!!!!

# Example

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
