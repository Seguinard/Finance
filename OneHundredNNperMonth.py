import pandas as pd
import datetime

import downloadSeriesAndGenerateAnalytics as dsaga
pd.set_option('display.max_rows', 100)


#I - Generate data
def generateAnalytics():
	#Used to generate the data for the mega validation with 100 neural networks per month
	pd.set_option('display.max_columns', None)
	#Get all tickers from S&P500
	folder = 'C:\\Users\\33695\\Desktop\\QIS_Data'
	filename = '100NeuralNetworksPerMonths'
	analytics = dsaga.analyticsGenerator(
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


#II - Find dates on which to build models
def check_if_last_day_of_month(date):
	import calendar
	#  calendar.monthrange return a tuple (weekday of first day of the 
	#  month, number  
	#  of days in month)
	last_day_of_month = calendar.monthrange(date.year, date.month)[1]
	# here i check if date is last day of month
	if date == datetime.date(date.year, date.month, last_day_of_month):
		return True
	return False


def create_last_day_of_month_mask(dateList):
	lastDayOfMonthMask=list()
	indexOfLastDays=list()
	for i in range(len(dateList)-1):
		if dateList[i].replace(day=1) != dateList[i+1].replace(day=1):
			lastDayOfMonthMask.append(True)
			indexOfLastDays.append(i)
		else:
			lastDayOfMonthMask.append(False)
	lastDayOfDataset = check_if_last_day_of_month(dateList[-1])
	lastDayOfMonthMask.append(lastDayOfDataset)
	if lastDayOfDataset:
		indexOfLastDays.append(len(dateList))
	return lastDayOfMonthMask, indexOfLastDays



analytics = pd.read_csv('C:\\Users\\33695\\Desktop\\QIS_Data\\100NeuralNetworksPerMonths.csv', index_col=0)

analytics.index = pd.to_datetime(analytics.index, format='%d/%m/%Y')


dates = analytics.index.to_list()

lastDayMask, lastDayIndex = create_last_day_of_month_mask(dates)

analytics['lastDayOfMonth'] = lastDayMask


print(analytics.tail(100))