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
	data_2017 = pandas.read_csv('GoogleTrendResults2017.csv',
						 names=['Title', 'Actor1', 'Actor2', 'Director', 'Studio', 'Share'])

	data_2018 = pandas.read_csv('GoogelTrendResults2018.csv',
						 names=['Title', 'Actor1', 'Actor2', 'Director', 'Studio', 'Share'])

	trainData2017, testData2017 = train_test_split(data_2017, test_size=0.2)
	trainData2018, testData2018 = train_test_split(data_2018, test_size=0.2)

	# Define Inputs and Outputs to NN - Feature=input, label=output
	trainLabels2017 = trainData2017.as_matrix(columns=['Share'])
	trainData2017 = trainData2017.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	testLabels2017 = testData2017.as_matrix(columns=['Share'])
	testData2017 = testData2017.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	np.random.shuffle(trainData2017)

	trainLabels2018 = trainData2018.as_matrix(columns=['Share'])
	trainData2018 = trainData2018.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	testLabels2018 = testData2018.as_matrix(columns=['Share'])
	testData2018 = testData2018.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	np.random.shuffle(trainData2018)

	#Neural Network based on 2017 Data
	model = build_model(trainData2017)

	#Saved to a file (Once)
	#model.save_weights('./mahWeights')

	#Training Model
	history = model.fit(trainData2017, trainLabels2017, epochs=1000, validation_split=0.2,
						verbose=0, callbacks=[PrintDot()])

	#Check performance
	#[loss, mae] = model.evaluate(testData, testLabels, verbose=0)
	#print("\nTesting set Mean Abs Error: ", mae)
	#plot_history(history)

	#Test Predictions
	testPredictions2017 = model.predict(testData2017)
	testPredictions2018 = model.predict(testData2018)

	#Error Difference Check
	error2017 = testPredictions2017 - testLabels2017
	#plt.hist(error2017)
	#plt.xlabel("Prediction Error")
	#_ = plt.ylabel("Count")

	#Predict 2018 shares
	error2018 = testPredictions2018 - testData2018
	# plt.show()
