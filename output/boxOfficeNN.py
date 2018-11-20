from __future__ import absolute_import, division, print_function
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)	

def build_model():
	model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(trainData.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)])

  	optimizer = tf.train.RMSPropOptimizer(0.001)

  	model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  	
	return model

def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [1000$]')
	plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
	plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
	plt.legend()
	print(np.array(history.history['val_mean_absolute_error']))
	plt.show()

class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='')

if __name__== "__main__":

	#Read in CSV with pandas
	df = pandas.read_csv('GoogleTrendResults2017.csv',
		names=['Title', 'Actor1', 'Actor2', 'Director', 'Studio', 'Share'])

	#print(df)
	trainData, testData = train_test_split(df, test_size=0.2)
	
	#Define Inputs and Outputs to NN - Feature is input, label is output
	trainLabels = trainData.as_matrix(columns=['Share'])
	trainData = trainData.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
	
	testLabels = testData.as_matrix(columns=['Share'])
	testData = testData.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])

	np.random.shuffle(trainData)
	
	print("Training Data\n", trainData[0:10])

	model = build_model()
	model.summary()
	model.save_weights('./mahWeights')

	EPOCHS = 100

	# Store training stats
	history = model.fit(trainData, trainLabels, epochs=EPOCHS, validation_split=0.2, verbose=0,
		callbacks=[PrintDot()])
	plot_history(history)
