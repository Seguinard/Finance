#This model is fed the data generated by the script 'downloadSeriesAndGenerateAnalytics.py'
#it will try to generate a model to find what asset to invest in on the next day for a holding period of slide day

#usual import
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Sklearn module
from sklearn.model_selection import train_test_split
#Keras module
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model


class neuralNetwork():
#Assign parameters to properties
	def __init__(self,
		nAsset, 
		nEpoch=10, nDenseDropoutLayer=10, dropoutRate=0.2, nWidthDenseLayer=100,
		showLearningCurve=False):
		self.nAsset=nAsset
		self.nEpoch=nEpoch
		self.nDenseDropoutLayer=nDenseDropoutLayer
		self.dropoutRate=dropoutRate
		self.nWidthDenseLayer=nWidthDenseLayer
		self.showLearningCurve=showLearningCurve
#The actual classification model
	def __classification_model(self, num_inputs, num_classes):
		model = Sequential()
		for i in range(self.nDenseDropoutLayer):
			model.add(Dropout(self.dropoutRate, input_dim=num_inputs))
			model.add(Dense(self.nWidthDenseLayer, activation='relu', kernel_constraint=tf.keras.constraints.max_norm(3)))
		model.add(Dropout(self.dropoutRate, input_dim=num_inputs))
		model.add(Dense(num_classes, activation='softmax'))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		return model
#A function to plot the learning curve and loss
	def ___plotModelLearningCurve(self):
		# list all data in history
		print(self.history.history.keys())
		# summarize history for accuracy
		plt.plot(self.history.history['accuracy'])
		plt.plot(self.history.history['val_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# summarize history for loss
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		return 
#Will create the model and save it into a property
	def build(self, pathToTrainSet):
		#Load data
		self.pathToTrainSet = pathToTrainSet
		financial_data = pd.read_csv(self.pathToTrainSet)  
		financial_data_cols = financial_data.columns
		#Get predictors
		predictors = financial_data[financial_data.columns[:len(financial_data.columns)-self.nAsset]]
		num_inputs = predictors.shape[1]
		#Get targets
		target = financial_data.drop(financial_data.columns[:-self.nAsset], axis=1)
		num_classes = target.shape[1]
		#Fix the Date column to week number instead of date
		predictors['Date']= pd.to_datetime(predictors['Date'])
		predictors['Date'] = predictors['Date'].dt.isocalendar().week
		#Normalize data
		predictors_norm = (predictors - predictors.mean()) / predictors.std()
		#Train test split
		predictors_train, predictors_test, target_train, target_test = train_test_split(predictors_norm, target, test_size=0.3)
		#Load the model
		model = self.__classification_model(num_inputs, num_classes)
		#Fit the model and saves history
		self.history = model.fit(predictors_train, target_train, validation_data=(predictors_test, target_test), epochs=self.nEpoch, verbose=1)
		#Plot Learning Curve
		if self.showLearningCurve:
			self.__plotModelLearningCurve(history)
		#Save to properties and return
		self.model = model
		return model
#Will predict on the validation set
	def validate(self, pathToValidationSet):
		if not hasattr(self, 'model'):
			raise AttributeError("You need to build the model with self.build() to be able to validate it")
		#Load data
		self.pathToValidationSet = pathToValidationSet
		financial_data = pd.read_csv(self.pathToValidationSet)  
		#Get predictors
		predictors = financial_data[financial_data.columns[:len(financial_data.columns)-self.nAsset]]
		num_inputs = predictors.shape[1]
		#Fix the Date column to week number instead of date
		predictors['Date']= pd.to_datetime(predictors['Date'])
		predictors['Date'] = predictors['Date'].dt.isocalendar().week
		#Normalize data
		predictors_norm = (predictors - predictors.mean()) / predictors.std()
		#Use the model to predict
		output=self.model.predict(predictors_norm)
		#Format output
		output = pd.DataFrame(output)
		output.columns = financial_data.columns[1:self.nAsset].to_list() + ['NUL']
		output.columns = [x + 'pick' for x in output.columns]
		output.index = financial_data['Date']
		dataForValidation = financial_data[financial_data.columns[1:self.nAsset]]
		dataForValidation.index = financial_data['Date']
		output = dataForValidation.merge(output, left_index=True, right_index=True)
		#Save to properties and return
		self.validationOutput = output
		return output
#Will save the prediction from self.validate
	def savePrediction(self, pathForBacktest):
		if not hasattr(self, 'validationOutput'):
			raise AttributeError("You need to validate the model with self.validate() to be able save the prediction from the validation")
		self.pathForBacktest = pathForBacktest
		self.validationOutput.to_csv(self.pathForBacktest,index=True)
		return



def main():
	#No argument = build the model for three assets and 10 epochs
	path = 'C:\\Users\\33695\\Desktop\\QIS_Data\\OutputSlideDayTargetToPredictWhatToPick'
	pathToTrainSet = path + 'TrainSet.csv'
	pathToValidationSet = path + 'ValidationSet.csv'
	pathForBacktest = path + 'ValidationToBacktest.csv'
	if len(sys.argv) != 3:
		nAsset=3
		nEpoch=10
	#2 argument = assets and epochs
	else:
		nAsset=int(sys.argv[1])
		nEpoch=int(sys.argv[2])
	model = neuralNetwork(
		nAsset,
		nEpoch=nEpoch, 
		nDenseDropoutLayer=10, 
		dropoutRate=0.2, 
		nWidthDenseLayer=100,
		showLearningCurve=False)
	model.build(pathToTrainSet)
	model.validate(pathToValidationSet)
	model.savePrediction(pathForBacktest)



if __name__ == '__main__':
	main()