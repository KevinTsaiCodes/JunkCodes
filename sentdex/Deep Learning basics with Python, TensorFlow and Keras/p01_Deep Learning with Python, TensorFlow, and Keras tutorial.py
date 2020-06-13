import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	print(tf.__version__)
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	print(x_train[0])
	"""
	It's generally a good idea to "normalize" your data. 
	This typically involves scaling the data to be between 0 and
	1, or maybe -1 and positive 1. In our case, each "pixel" is a
	feature, and each feature currently ranges from 0 to 255. 
	0 Not quite 0 to 1. Let's change that with a handy utility function:
	"""
	x_train = tf.keras.utils.normalize(x_train, axis=1)
	x_test = tf.keras.utils.normalize(x_test, axis=1)
	print('\nAfter Scale')
	print(x_train[0])
	model = tf.keras.models.Sequential() # create model function
	model.add(tf.keras.layers.Flatten()) # input layer
	model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # hidden layer
	# model.add(tf.keras.layers.Dense(neurals,activation=tf.nn.activate_functions))
	model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) # hidden layer
	model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # output layer
	
	# training
	model.compile(optimizer='adam', 
	              loss='sparse_categorical_crossentropy', # get accuary
	              metrics=['accuracy'])
	"""
	Now we need to "compile" the model. This is where we pass the settings
	for actually optimizing/training the model we've defined.
	"""
	model.fit(x_train, y_train, epochs=3) # train your model
	# A neural network doesn't actually attempt to maximize accuracy. It attempts to minimize loss.
