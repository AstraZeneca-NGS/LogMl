
# Introduction

Log(ML) is a framework that helps automate many steps in machine learning projects and let you quickly generate baseline results.

**Why?**
There is a considerable amount is setup, boiler-plate code, analysis in every ML/AI project.
`Log(ML)` performs most of these boring tasks, so you can focus on what's important and adds value.

`Log(ML)` performs a consistent data science pipeline, keeping track every action and saving all results and models automatically.

**Log(ML) Goals: What does Log(ML) do for me?**
Log(ML) is designed to:
- Enforce best practices
- Perform a set of common, well defined, well tested analyses
- Quickly turn around the first analysis results
- Facilitates logging in ML projects: No more writing down results in a notepad, `Log(ML)` creates log file in a systematic manner
- Save models and results: `Log(ML)` saves all your models, so you can always retrieve the best ones.

**Architecture: How does Log(ML) work?**
`Log(ML)` has a standard "data science workflow" (a.k.a. pipeline).
The workflow include several steps, such as data preprocessing, data augmentation, data exploration, feature importance, model training, hyper-parameter search, cross-validation, etc.
Each step in the workflow can be customized either in a configuration YAML file or adding custom Python code.

# Install

Requirements:
- Python 3.7
- Virtual environment

```
git clone https://github.com/AstraZeneca-NGS/LogMl.git

cd LogMl
./scripts/install.sh
```

The `scripts/install.sh` script should take care of installing in a default directory (`$HOME/logml`).
If you want another directory, just edit the script and change the `INSTALL_DIR` variable

# Nomenclature

Parameters from YAML: We refer to parameters defined in YAML file as between curly brackets, e.g. `{parameter_name}`

User defined functions: This are functions defined by the user and marked with the `Log(ML)` annotations. For instance, the "user function decorated with `@dataset_load`" is sometimes referred as the "`@dataset_load` function", for short
