from __future__ import absolute_import, division, print_function
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas
import warnings
import sys
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
	#First Run: Trains neural network, saving weights of the model
	#Sequential Runs: Load in 2018/any dataset to predict its shares

	#READ INPUT DATA
	data_2017 = pandas.read_csv(sys.argv[1],
						 names=['Title', 'Actor1', 'Actor2', 'Director', 'Studio', 'Share'])



	#SPLIT DATA INTO TRAINING AND TESTING SETS
	trainData2017, testData2017 = train_test_split(data_2017, test_size=0.2)


	# Define Inputs and Outputs to NN - Feature=input, label=output
	trainLabels2017 = trainData2017.as_matrix(columns=['Share'])
	trainData2017 = trainData2017.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	testLabels2017 = testData2017.as_matrix(columns=['Share'])
	testData2017 = testData2017.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	np.random.shuffle(trainData2017)

	#BUILD NEURAL NETWORK
	model = build_model(trainData2017)


	#2017 DATA - TRAINING MODEL
	history2017 = model.fit(trainData2017, trainLabels2017, epochs=1000, validation_split=0.2,
						verbose=0, callbacks=[PrintDot()])

	model.save_weights('./2017weights')
	#2017 DATA - SAVE WEIGHTS ONCE

	#2017 DATA - CHECK PERFORMANCE
	#[loss, mae] = model.evaluate(testData2017, testLabels2017, verbose=0)
	#print("\nTesting set Mean Abs Error: ", mae)
	#plot_history(history)

	#2017 DATA - TEST PREDICTIONS
	testPredictions2017 = model.predict(testData2017)

	#2017 DATA - PREDICTIONS VS GIVEN SHARES
	error2017 = testPredictions2017 - testLabels2017
	plt.hist(error2017)
	plt.xlabel("2017 Shares Prediction Error")
	_ = plt.ylabel("Count")
	plt.show()
