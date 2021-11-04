import downloadSeriesAndGenerateAnalytics as dsaga
import kerasClassificationFromGeneratedAnalytcis as ker
import backtestBasket as bb
import pandas as pd



def main():
	#Parameters #Commented out: best parameters found for skewness target
	NeedToRegenerateAnalytics=True

	#Portfolio Rebalancing
	rebalancePeriod=21 #21
	rebalanceProportion=None #None
	holdingPeriod=21 #21
	multiAssetPick=False #False

	#Portfolio and target - ETFs: ['IXC','IXG','IXJ','IXN','RXI','JXI','EXI','KXI','REET','AAXJ','EEM','IVV','IEUR','EWJ','GLD']
	tickerList=  ['SSO', 'SGOL'] #['SSO', 'SGOL']
	start="2012-01-01" #"2012-01-01"
	end=None
	slide=holdingPeriod #We want to match our prediction target to the time we hold

	#Neural Network Training #Commented out: best parameters found for skewness target
	target='SegSkewness' #SegSkewness
	memory=0 #0
	trainValidationSplit=1/2 #1/2
	nEpoch=100 #100
	nDenseDropoutLayer=9 #9
	dropoutRate=0.2 #0.2
	nWidthDenseLayer=100 #100
	showLearningCurve=NeedToRegenerateAnalytics #Display it when changed stuff and so need to regenerate anaytics 

	#Others
	nulPickWeight=True
	weightsComputedAtEOD=True
	
	leverage=1
	precisionExponent=6
	startingValue=100

	folder = 'C:\\Users\\33695\\Desktop\\QIS_Data'
	filename = 'OutputSlideDayTargetToPredictWhatToPick'
	
	#dsaga
	if NeedToRegenerateAnalytics:
		print("-"*50)
		print("Generating Data Set")
		print("-"*50)
		analytics = dsaga.analyticsGenerator(
			tickerList, 
			start=start, 
			end=end, 
			target=target, 
			slide=slide,  
			memory=memory,
			trainValidationSplit=trainValidationSplit)
		analytics.generate()
		analytics.save(folder, filename)
	else:
		print("-"*50)
		print("Using precedently generated Data Set")
		print("-"*50)
	
	#ker
	print("-"*50)
	print("Training Model")
	print("-"*50)
	pathToTrainSet = folder + '\\' + filename + 'TrainSet.csv'
	pathToValidationSet = folder + '\\' + filename + 'ValidationSet.csv'
	pathForBacktest = folder + '\\' + filename + 'ValidationToBacktest.csv'
	model = ker.neuralNetwork(
		len(tickerList)+1,
		nEpoch=nEpoch, 
		nDenseDropoutLayer=10, 
		dropoutRate=0.2, 
		nWidthDenseLayer=100,
		showLearningCurve=False)
	model.build(pathToTrainSet)
	model.validate(pathToValidationSet)
	model.savePrediction(pathForBacktest)

	
	#bb
	print("-"*50)
	print("Backtesting Model")
	print("-"*50)
	data = pd.read_csv(pathForBacktest, index_col=0)
	backtest = bb.assetBasket(
		data, 
		nulPickWeight=nulPickWeight,
		 weightsComputedAtEOD=weightsComputedAtEOD)
	bt = backtest.backtest(
		rebalancePeriod=rebalancePeriod, 
		rebalanceProportion=rebalanceProportion, 
		holdingPeriod=holdingPeriod, 
		multiAssetPick=multiAssetPick, 
		leverage=leverage,
		precisionExponent=precisionExponent, 
		startingValue=startingValue)
	backtest.save(path=folder + '\\')

	print("-"*50)
	print("Backtest Ready - Final Value: {f:.2f} on {d}".format(f=bt.iloc[-1,-1], d=bt.index.to_list()[-1]))
	print("Final Alocation:")
	print(backtest.historical_portfolio.iloc[-1:,])
	print("-"*50)

	import matplotlib.pyplot  as plt
	bt['backtest'].plot()
	plt.show()
	#backtester.historical_portfolio.plot.bar(stacked=True)
	#plt.show()



if __name__ == '__main__':
	main()

