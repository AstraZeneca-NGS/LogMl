

# LogMl

![LogMl](img/logml_logo.png)

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
The workflow include several steps, such as data transformation, data augmentation, data exploration, feature importance, model training, hyper-parameter search, cross-validation, etc.
Each step in the workflow can be customized either in a configuration YAML file or adding custom Python code.

![LogMl](img/LogMl.png)
