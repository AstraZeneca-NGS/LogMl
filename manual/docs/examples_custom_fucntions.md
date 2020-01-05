
# Learning by examples

This is Machine Learning, so let's learn by showing some examples...(hopefully you can generalize)

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
