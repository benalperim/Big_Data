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

if __name__ == "__main__":
    data_2018 = pandas.read_csv(sys.argv[1],
                                names=['Title', 'Actor1', 'Actor2', 'Director', 'Studio', 'Share'])

    trainData2018, testData2018 = train_test_split(data_2018, test_size=0.2)

    trainLabels2018 = trainData2018.as_matrix(columns=['Share'])
    trainData2018 = trainData2018.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])
    testLabels2018 = testData2018.as_matrix(columns=['Share'])
    testData2018 = testData2018.as_matrix(columns=['Title', 'Actor1', 'Actor2', 'Director', 'Studio'])

    combinedData2018 = np.concatenate([trainData2018, testData2018])
    combineShares2018 = np.concatenate([trainLabels2018, testLabels2018])
    np.random.shuffle(combinedData2018)

    model = build_model(trainData2018)
    model.load_weights('./2017weights')

    testPredictions2018 = np.absolute(model.predict(combinedData2018))
    df = pandas.DataFrame(combinedData2018)
    df_shares = pandas.DataFrame(testPredictions2018)
    df['Shares'] = df_shares
    print(df.shape)
    df.to_csv('predicted_2018_shares.csv', sep=",", index=False, header=None)

    #2018 DATA - MEAN AVERAGE ERROR
    [loss, mae] = model.evaluate(testData2018, testLabels2018, verbose=0)
    print("Testing set Mean Abs Error: {:1.5f}".format(mae))

    error2018 = testPredictions2018 - combineShares2018
    plt.hist(error2018)
    plt.xlabel("2018 Shares Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()
