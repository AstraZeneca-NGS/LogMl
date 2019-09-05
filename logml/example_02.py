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
	return model.fit(dataset[:, 0:2], dataset[:, 2], epochs=epochs, verbose=0)


@model_evaluate
def my_model_eval(model, dataset):
	return model.evaluate(dataset[:, 0:2], dataset[:, 2], verbose=0)


lm = LogMl()
lm()
