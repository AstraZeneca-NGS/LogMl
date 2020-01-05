
# Hyper-parameter optimization

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

### Examples

Perform hyper-parameter optimization of the learning rate using a uniform distribution as a p
```
hyper_parameter_optimization:
    algorithm: 'tpe'
    max_evals: 100
    space:
        model_train:
            learning_rate: ['uniform', 0.0, 0.5]
```

Hyper parameter optimization not only can be used with `model_train`, but also with other *named_workflow_steps*, such as `model_create`

```
hyper_parameter_optimization:
    algorithm: 'tpe'
    max_evals: 100
    space:
        model_create:
            layer_1: ['randint', 10]
        model_train:
            learning_rate: ['uniform', 0.0, 0.5]
```

You can even perform hyper-parameter optimization including the `dataset_preprocess` *named_workflow_step*
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
