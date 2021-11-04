#Imports
import financialTimeSeriesAnalytics as fta
import numpy as np
import pandas as pd
import sys
from datetime import datetime


#This class will download time series from yahoo fiannce
#Then it will slide analytics to create predictors for a model 
#It will then set a target for the model to predict
#The output can be saved or stored in variables
class analyticsGenerator:
	def __init__(self, 
		tickerList, start='2000-01-01', end=None, 
		target='SegSkewness', slide=21,  memory=0,
		trainValidationSplit=1/2):
		#First we assign properties
		self.start=start
		self.end=end
		self.tickerList=tickerList
		self.target=target
		self.slide=slide
		self.memory=memory
		self.trainValidationSplit=trainValidationSplit
		self.nonSlidedmetrics=[['basic', 'absoluteReturn()',1,21],
			   				   ['basic', 'absoluteReturn()',1,126],
			   				   ['basic', 'absoluteReturn()',1,252],
			   				   ['basic', 'volatility()',1,21],
			   				   ['basic', 'volatility()',1,126],
			   				   ['basic', 'volatility()',1,252],
			   				   ['distribution', 'skewness()',1,21],
			   				   ['distribution', 'skewness()',1,126],
			   				   ['distribution', 'skewness()',1,252]]
#Will check there is a end date and generate one if not
	def __checkEndDate(self):
		if self.end is None:
			self.end = datetime.today().strftime('%Y-%m-%d')
#Will download required data and store in in a matrix as well as save all fta objects in a list
	def __downloadAndAggregateData(self):
		assetsList = list()
		output = pd.DataFrame()
		i = -1
		for ticker in self.tickerList:
			i+=1
			#Download data
			try:
				currentAsset = fta.importDataTimeSeries(ticker, self.start, self.end)
				assetsList.append(currentAsset)
				#a) initialize with time Series
				if len(assetsList) == 1:
					output = pd.DataFrame({'Date': currentAsset.seriesDates, '{}'.format(self.tickerList[i]): currentAsset.timeSeries})
				else:
					temp = pd.DataFrame({'Date': currentAsset.seriesDates, '{}'.format(self.tickerList[i]): currentAsset.timeSeries})
					output = output.merge(temp, on='Date', how='left')
			except Exception:
				self.tickerList.remove(ticker)
		self.assetsList = assetsList
		self.analyticsMatrix = output
		return 
#Will generate the columns with the value of each asset up to memory days in the past (so if memory=0 will not generate anything)
	def __generateMemoryColumns(self):
		if self.memory > 0:
			output = self.analyticsMatrix
			for ticker in self.tickerList:
				print(ticker + 'memory1-' + str(self.memory))
				for i in range(1, self.memory + 1):
					colName = ticker + '_day-{}'.format(i)
					output[colName] = output[ticker].pct_change(1).shift(i)
			self.analyticsMatrix =  output
#Will generate the indicators that are naturally slided indicators and so do not need to be slided
	def __generateSlidedIndicators(self):
		output = self.analyticsMatrix
		i = -1
		for currentAsset in self.assetsList:
			i+=1
			temp = pd.DataFrame({'Date': currentAsset.seriesDates, '{}_rsi'.format(self.tickerList[i]): currentAsset.technical.relativeStrengthIndex()})
			output = output.merge(temp, on='Date', how='left')
			temp = pd.DataFrame({'Date': currentAsset.seriesDates, '{}_macd'.format(self.tickerList[i]): currentAsset.technical.movingAverageConvergenceDivergence()})
			output = output.merge(temp, on='Date', how='left')
		self.analyticsMatrix =  output
		return 
	def __generateNonSlidedIndicators(self):
		output = self.analyticsMatrix
		metrics = self.nonSlidedmetrics
		i = -1
		for asset in self.assetsList:
			i+=1
			name = self.tickerList[i]
			for metric in metrics:
				print(name + metric[1] + str(metric[3]))
				temp = asset.slide_analytic(
					analyticClass=metric[0], 
					analyticName=metric[1], 
					slidingPeriod=metric[2],
					calculationPeriod=metric[3],
					startCalculationAt=0,
					succinctness=10000,
					nameToPut=name)
				output = output.merge(temp, on='Date', how='left')
		self.analyticsMatrix =  output
		return
	def __generatePredictionMetrics(self):
		if self.target == 'Return':
			predMetrics = [['basic', 'absoluteReturn()',1,self.slide]]
		elif self.target == 'Skewness':
			predMetrics = [['distribution', 'skewness()',1,self.slide]]
		elif self.target == 'SharpeRatio':
			predMetrics = [['basic', 'absoluteReturn()',1,self.slide],
					       ['basic', 'volatility()',1,self.slide]]
		elif self.target == 'CalmarRatio':
			predMetrics = [['basic', 'absoluteReturn()',1,self.slide],
					   	   ['drawDown', 'max()',1,self.slide]]
		elif self.target == 'SegRatio':
			predMetrics = [['basic', 'absoluteReturn()',1,self.slide],
					       ['basic', 'volatility()',1,self.slide],
					       ['distribution', 'skewness()',1,self.slide]]
		elif self.target == 'DrawDownSegRatio':
			predMetrics = [['basic', 'absoluteReturn()',1,self.slide],
					       ['basic', 'volatility()',1,self.slide],
					       ['distribution', 'skewness()',1,self.slide],
					       ['drawDown', 'max()',1,self.slide]]
		elif self.target == 'SegSkewness':
			predMetrics = [['distribution', 'skewness()',1,self.slide]]
		else:
			raise ValueError("Supported targets are: 'Return', 'Skewness', 'SharpeRatio', 'CalmarRatio', 'SegRatio', 'DrawDownSegRatio'")
		self.predMetrics = predMetrics
		return 
	def __generatePreTarget(self, nameNewCol, ticker):
		target = self.target
		slide = self.slide
		output = self.analyticsMatrix
		if target == 'Return':
			output[nameNewCol] = output['pred{t}absoluteReturn(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]
		elif target == 'Skewness':
			output[nameNewCol] = output['pred{t}skewness(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]
		elif target == 'SharpeRatio':
			output[nameNewCol] = (output['pred{t}absoluteReturn(){f:.0f}'.format(t=ticker, f=slide)].iloc[:] 
									/ 
								  output['pred{t}volatility(){f:.0f}'.format(t=ticker, f=slide)].iloc[:])
		elif target == 'CalmarRatio':
			output[nameNewCol] = (output['pred{t}absoluteReturn(){f:.0f}'.format(t=ticker, f=slide)].iloc[:] 
									/ 
								  (- output['pred{t}max(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]))
		elif target == 'SegRatio':
			output[nameNewCol] = ((output['pred{t}absoluteReturn(){f:.0f}'.format(t=ticker, f=slide)].iloc[:] 
									/ 
								   output['pred{t}volatility(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]) 
									* 
								 ( 1 + output['pred{t}skewness(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]))
		elif target == 'DrawDownSegRatio':
			output[nameNewCol] = ((output['pred{t}absoluteReturn(){f:.0f}'.format(t=ticker, f=slide)].iloc[:] 
									/ 
								   output['pred{t}volatility(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]) 
									* 
								 ( 1 + output['pred{t}skewness(){f:.0f}'.format(t=ticker, f=slide)].iloc[:])
									*
								 ( 1 + output['pred{t}max(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]))
		elif target == 'SegSkewness':
			output[nameNewCol] =  1 + output['pred{t}skewness(){f:.0f}'.format(t=ticker, f=slide)].iloc[:]
		else:
			raise ValueError("Supported targets are: 'Return', 'Skewness', 'SharpeRatio', 'CalmarRatio', 'SegRatio', 'DrawDownSegRatio', 'Mysterious'")
		return output
	def __generateTarget(self):
		output = self.analyticsMatrix
	#a) 'slide' days factors that will be used To predict
		self.__generatePredictionMetrics()
		predList = list()
		i = -1
		for asset in self.assetsList:
			i+=1
			name = 'pred' + self.tickerList[i]
			for metric in self.predMetrics:
				print(name + metric[1] + str(metric[3]))
				temp = asset.slide_analytic(
					analyticClass=metric[0], 
					analyticName=metric[1], 
					slidingPeriod=metric[2],
					calculationPeriod=metric[3],
					startCalculationAt=0,
					succinctness=10000,
					nameToPut=name)
				predList.append(temp.columns[1])
				output = output.merge(temp, on='Date', how='left')
		self.analyticsMatrix = output
	#b) create the pre target
		newColList = list()
		for ticker in self.tickerList:
			nameNewCol = 'slideDayPredFactor{f}{t}'.format(f=self.target,t=ticker)
			newColList.append(nameNewCol)
			output = self.__generatePreTarget(nameNewCol, ticker)
	#c) create the target from pretarget
		pickColList = list()
		i = -1
		for newCol in newColList:
			i+=1
			namePickCol = self.tickerList[i] + 'pick'
			pickColList.append(namePickCol)
			newColListTemp = newColList[:]
			newColListTemp.remove(newCol)
			output[namePickCol] = output.apply(lambda x: 1 if x[newCol] > np.max(x[newColListTemp]) and x[newCol] >= 0 else 0, axis = 1)
		output['NULpick'] = output.apply(lambda x: 1 if np.sum(x[pickColList]) == 0 else 0, axis = 1)
	#d) Drop now useless columns
		output = output.drop(newColList, axis=1)
		output = output.drop(predList, axis=1)
	#e) perform the shift for prediction
		for pickCol in pickColList:
			output[pickCol] = output[pickCol].shift(-self.slide)
		output['NULpick'] = output['NULpick'].shift(-self.slide)
	#f) return
		self.analyticsMatrix = output
		return
	def __splitTrainTestAndValidation(self):
		self.trainSetClassification = self.analyticsMatrix[:int(len(self.analyticsMatrix)*self.trainValidationSplit)]
		self.validationSetClassification = self.analyticsMatrix[int(len(self.analyticsMatrix)*self.trainValidationSplit):]
	#Only keep train set data starting from row where we have all data
		self.trainSetClassification = self.trainSetClassification.dropna()
		return
	def generate(self):
	#a) Check if there is a end date and puts one if not
		self.__checkEndDate()
	#b) get data
		self.__downloadAndAggregateData()
	#c) generate columns of asset level memory
		self.__generateMemoryColumns()
	#d) generate indicators that are naturally slided so no need to slide them
		self.__generateSlidedIndicators()
	#e) generate indicators that are not naturally slided so need to slide them
		self.__generateNonSlidedIndicators()
	#f) generate the target
		self.__generateTarget()
	#g) Create two sets
		self.__splitTrainTestAndValidation()
		return self.analyticsMatrix
	def save(self, folder='', filename=None, trainSet=True, validationSet=True, 
		analyticsMatrix=False):
		if filename is None:
			filename = 'OutputSlideDayTargetToPredictWhatToPick'
		path = folder + '\\' + filename 
		if trainSet:
			self.trainSetClassification.to_csv(path +'TrainSet.csv', index =False)
			print('Saved train set file as {}'.format(path +'TrainSet.csv'))
		if validationSet:
			self.validationSetClassification.to_csv(path + 'ValidationSet.csv', index =False)
			print('Saved validation set file as {}'.format(path +'ValidationSet.csv'))
		if analyticsMatrix:
			self.analyticsMatrix.to_csv(path + 'analyticsMatrix.csv', index =False)
			print('Saved analyticsMatrix file as {}'.format(path +'analyticsMatrix.csv'))
		return



def main():
	#Used to generate the data for the mega validation with 100 neural networks per month
	pd.set_option('display.max_columns', None)
	#Get all tickers from S&P500
	folder = 'C:\\Users\\33695\\Desktop\\QIS_Data'
	filename = '100NeuralNetworksPerMonths'
	analytics = analyticsGenerator(
		['SSO', 'SGOL'], 
		start='2009-09-09', 
		end=None, 
		target='SegSkewness', 
		slide=21, 
		memory=0,
		trainValidationSplit=1/2)
	analyticsGenerated = analytics.generate()
	#Clean the rows with nan in begining of file
	analyticsGenerated = analyticsGenerated.iloc[253:,:]
	analyticsGenerated.to_csv(folder + '\\' + filename + '.csv', index=False)



if __name__ == '__main__':
	main()