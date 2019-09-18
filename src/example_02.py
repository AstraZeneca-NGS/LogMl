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


@dataset_inout
def my_dataset_inout(dataset):
	return dataset[:, 0:2], dataset[:, 2]


@model_create
def my_model_create(x, y, num_neurons):
	model = Sequential()
	model.add(Dense(num_neurons, activation='tanh', input_shape=(2,)))
	model.add(Dense(1, activation='tanh'))
	return model


@model_train
def my_model_train(model, x, y, learning_rate, epochs):
	model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error')
	return model.fit(x, y, epochs=epochs, verbose=0)


@model_evaluate
def my_model_eval(model, x, y):
	return model.evaluate(x, y, verbose=0)


lm = LogMl()
lm()
