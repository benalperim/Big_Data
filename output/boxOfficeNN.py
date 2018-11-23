from __future__ import absolute_import, division, print_function
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def build_model(trainData):
	model = keras.Sequential([
		keras.layers.Dense(64, activation=tf.nn.sigmoid,
						   input_shape=(trainData.shape[1],)),
		keras.layers.Dense(64, activation=tf.nn.sigmoid),
		keras.layers.Dense(1)
	])

	optimizer = tf.train.RMSPropOptimizer(0.000001)

	model.compile(loss='mse',
				  optimizer=optimizer,
				  metrics=['mae'])

	return model


def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [1000$]')
	plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
			 label='Training Loss')
	plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
			 label='Valid loss')
	plt.legend()
	plt.show()


class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print(':', end='')


if __name__ == "__main__":
	df = pandas.read_csv('GoogleTrendResults2017.csv',
						 names=['Title', 'Actor1', 'Actor2', 'Director', 'Studio', 'Share'])

	trainData, testData = train_test_split(df, test_size=0.2)

	# Define Inputs and Outputs to NN - Feature=input, label=output
	trainLabels = trainData.as_matrix(columns=['Share'])
	trainData = trainData.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	testLabels = testData.as_matrix(columns=['Share'])
	testData = testData.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])

	#Generic Shuffle
	np.random.shuffle(trainData)

	#NN Model
	model = build_model(trainData)

	#Saved to a file
	model.save_weights('./mahWeights')

	#Training Model
	history = model.fit(trainData, trainLabels, epochs=1000, validation_split=0.2,
						verbose=0, callbacks=[PrintDot()])

	#Check performance
	[loss, mae] = model.evaluate(testData, testLabels, verbose=0)
	print("\nTesting set Mean Abs Error: ", mae)

	plot_history(history)

	#Test Predictions
	testPredictions = model.predict(testData)

	#Error Difference
	error = testPredictions - testLabels
	plt.hist(error)
	plt.xlabel("Prediction Error")
	_ = plt.ylabel("Count")

	plt.show()
