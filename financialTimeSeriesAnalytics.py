######################################################################################################
######################################################################################################
###################################### IMPORT PACKAGES ###############################################
######################################################################################################
######################################################################################################

import numpy as np
import pandas as pd
import yfinance as yf
from math import sqrt, floor
from scipy.stats import skew, kurtosis
from datetime import date, datetime, timedelta


######################################################################################################
######################################################################################################
############################### PARAMETER - GLOBAL VARIABLE ##########################################
######################################################################################################
######################################################################################################

#The global variable to know the number of sub classes that exist
#Necessary to initialize financialTimeSeries Class
nSC = 7


######################################################################################################
######################################################################################################
################################### CODE - MAIN CLASS ################################################
######################################################################################################
######################################################################################################

#Class for financial time series data
#entryType = 'timeSeries' or entryType = dailyReturns
#seriesDates or seriesBenchmark can be added at generation or later through the appropriated method
class financialTimeSeries:
#1. Initilization
#Initilization of the class and restrictions
	def __init__(self, entry, entryType='dailyReturns', 
		entryDates=np.zeros(1),
		entryBenchmark=np.zeros(1),
		subclassesToInitialize=np.zeros(nSC)):
		#Initialize universal attributes
		self.mainAttributesAssigner(entry, entryType, 
			entryDates, entryBenchmark)
		#Initialize selected sub classes
		if hasattr(self, 'seriesBenchmark'):
			self.benchmarked()
		if subclassesToInitialize[0]:
			self.basic()
		if subclassesToInitialize[1]:
			self.distribution()
		if subclassesToInitialize[2]:
			self.drawDown()
		if subclassesToInitialize[3]:
			self.drawUp()
		if subclassesToInitialize[4]:
			self.timeToNewHigh()
		if subclassesToInitialize[5]:
			self.supportLine()
		if subclassesToInitialize[6]:
			self.technical()
#2. Subclasses
#Generate the subclass for basic analytics
	def basic(self):
		if hasattr(self, 'seriesDates'):
			self.basic = basic(self.timeSeries, self.seriesDates)
		else:
			self.basic = basic(self.timeSeries)
		return True
#Generate the subclass for benchmarked analytics
	def benchmarked(self):
		if not hasattr(self, 'seriesBenchmark'):
			raise AttributeError(
				'You must define seriesBenchmark attribute to use this subclass')
		if hasattr(self, 'seriesDates'):
			self.benchmarked = benchmarked(self.timeSeries, 
				self.seriesBenchmark, 
				self.seriesDates)
		else:
			self.benchmarked = benchmarked(self.timeSeries, 
				self.seriesBenchmark)
		return True
#Generate the subclass for distribution related analytics
	def distribution(self):
		self.distribution = distribution(self.dailyReturns)
		return True
#Generate the subclass for drawdown related analytics
	def drawDown(self):
		self.drawDown = drawDown(self.timeSeries)
		return True
#Generate the subclass for drawup related analytics
	def drawUp(self):
		self.drawUp = drawUp(self.timeSeries)
		return True
#Generate the subclass for time to new high related analytics
	def timeToNewHigh(self):
		self.timeToNewHigh = timeToNewHigh(self.timeSeries)
		return True
#Generate the subclass for support line related analytics
	def supportLine(self):
		if not hasattr(self, 'seriesDates'):
			raise AttributeError(
				'You must define seriesDates attribute to use this subclass')
		self.supportLine = supportLine(self.timeSeries, self.seriesDates)
		return True
#Generate the subclass for technical analysis related analytics
	def technical(self):
		self.technical = technical(self.timeSeries)
		return True
#3. Useful Generic Methods
#Assign series, daily returns, NElement, dates and benchmark
	def mainAttributesAssigner(self, entry, entryType, entryDates, entryBenchmark):
		#check that entry is np array
		if not type(entry) == np.ndarray : 
			raise TypeError('{} must be set to a ndarray'.format(entryType))
		#Generate the size attribute
		self.NElements = np.shape(entry)[0]
		#check if was fed dailyReturns or timeSeries
		if entryType == 'dailyReturns':
			self.dailyReturns = entry
			self.rebase()
		elif entryType == 'timeSeries': 
			self.timeSeries = entry
			self.dailyReturns = np.divide(self.timeSeries[1:],
				self.timeSeries[:-1]) - 1
		else:
			raise ValueError("entryType must be 'timeSeries' or 'dailyReturns'")
		#check if was fed dates
		if np.shape(entryDates) != np.shape(np.zeros(1)):
			self.seriesDatesAssigner(entryDates)
		#check if was fed benchmark
		if np.shape(entryBenchmark) != np.shape(np.zeros(1)):
			self.seriesBenchmarkAssigner(entryBenchmark)
		return True
#Assign dates to the financialTimeSeries
	def seriesDatesAssigner(self, dates):
		if not type(dates) == np.ndarray:
			raise TypeError('dates must be set to a ndarray')
		elif np.shape(dates)[0] != self.NElements:
			raise ValueError('dates must be the length of timeSeries')
		else:
			self.seriesDates = dates
#Assign benchmark to the financialTimeSeries
	def seriesBenchmarkAssigner(self, benchmark):
		if not type(benchmark) == np.ndarray:
			raise TypeError('benchmark must be set to a ndarray')
		elif np.shape(benchmark)[0] != self.NElements:
			raise ValueError('benchmark must be the length of timeSeries')
		else:
			self.seriesBenchmark = benchmark
#Rebase timeSeries, creates it from dailyReturns if none already existing
	def rebase(self, fees=0, dayConvention=365):
		N = np.shape(self.dailyReturns)[0]
		temp = np.ones(N)
		temp[0] = 100
		for k in range(N-1):
			temp[k+1] = temp[k] * (1 + self.dailyReturns[k] - fees/dayConvention)
		self.rebase = temp
		if not hasattr(self,'timeSeries'):
			self.timeSeries = self.rebase
		return temp
#Return corresponding Business Days number to a period
	def periodTypeMatcherToBD(self, periodType):
		if periodType == "d":
			return 1
		elif periodType == "w":
			return 5
		elif periodType == "m":
			return 21
		elif periodType == "y":
			return 252
#Return time difference between two time objects as integer
	def datediff(self, date1, date2, unit="d", dayConvention=365):
		if unit == "d":
			return int((date2-date1)/(10**9 * 60 * 60 *24))
		elif unit == "m":
			return int((date2-date1)/(10**9 * 60 * 60 *24)//(dayConvention/12))
		elif unit == "y":
			return int((date2-date1)/(10**9 * 60 * 60 *24)//dayConvention)
		else:
			raise ValueError("Please enter 'd', 'm' or 'y'")
#Initialize the sub class from which it is input the name
	def subClassToInitialize(self, className, array=np.zeros(nSC)):
		if np.shape(array) != np.shape(np.zeros(nSC)):
			raise ValueError("array must be of size ({},)".format(nSC))
		#Associate to correct position
		if className == "basic":
			i = 0 
		elif className == "distribution":
			i = 1
		elif className == "drawDown":
			i = 2
		elif className == "drawUp":
			i = 3
		elif className == "timeToNewHigh":
			i = 4
		elif className == "supportLine":
			i = 5
		elif className == "technical":
			i = 6
		#Initialize it to True
		array[i] = 1
		return array
#4. Analytic slider
#Will slide an analytic over the whole data available with a step equal to slidingPeriod
#Using the past calculationPeriod for each calculation as input
#Calculating every calculationPeriod from the beginning plus 1*calculationPeriod
	def slide_analytic(
		self, 
		analyticClass='basic', 
		analyticName='absoluteReturn', 
		slidingPeriod=1, 
		calculationPeriod=0,
		startCalculationAt=0,
		succinctness=100,
		nameToPut=""):
		#Retrieve series from the financialTimeSeriesObject
		try:
			timeSeries = self.timeSeries
			entryDates = self.seriesDates
		except AttributeError:
			raise AttributeError('You must define seriesDates attribute to use this method')
		#Find what class to initialize
		classBooleans = self.subClassToInitialize(analyticClass)
		#Iterate over data for slidingPeriod with calculationPeriod
		resList = list()
		datesList = list()
		counter = int(0)
		for i in range(calculationPeriod + startCalculationAt, np.shape(timeSeries)[0], slidingPeriod):
			counter += 1
			#Find what backLook to use
			#calculationPeriod == 0 means that we use all the information available upt to date
			if calculationPeriod == 0:
				backLook = i
			#calculationPeriod > 0 means that we use a constant size vector for the calculation
			else:
				backLook = calculationPeriod
			#Create the time series with desired class module implemented
			timeSeriesToUse = timeSeries[i-backLook:i + 1]
			entryDatesToUse = entryDates[i-backLook:i + 1]
			newFinancialTimeSeriesObject = financialTimeSeries(
				timeSeriesToUse, 
				entryType='timeSeries',
				entryDates=entryDatesToUse,
				subclassesToInitialize=classBooleans)
			#Executes the custom string as code
			res = 0
			toExecute = "res = newFinancialTimeSeriesObject.{}.{}".format(analyticClass, analyticName)
			loc = {"newFinancialTimeSeriesObject":newFinancialTimeSeriesObject,"res":res}
			exec(toExecute, globals(), loc) #executes the str toExecute as code and returns in loc
			res = loc['res']
			resList.append(res)
			datesList.append(entryDates[i])
			if counter / succinctness == counter//succinctness:
				print(str(entryDates[i]) + " - " + str(res))
		#Generate output
		title = nameToPut + analyticName + str(calculationPeriod)
		output = pd.DataFrame({"Date":datesList, title:resList})
		return output


######################################################################################################
######################################################################################################
################################# CODE - BASIC SUBCLASS ##############################################
######################################################################################################
######################################################################################################

#Child Class of financialTimeSeries used for basic analytics
#entryType = 'timeSeries'
class basic(financialTimeSeries):
#Initilization of the class and restrictions
	def __init__(self, timeSeries, seriesDates=np.zeros(1)):
		#Initialize the father class
		financialTimeSeries.__init__(self, timeSeries, entryType='timeSeries',
			entryDates=seriesDates)
#Calculate square root of 2nd moment = volatility
	def volatility(self, period=252):
		temp = np.std(np.log(self.dailyReturns+1)) * sqrt(period)
		self.volatility = temp
		return temp
#Calculate absolute return
	def absoluteReturn(self):
		temp = self.timeSeries[-1]/self.timeSeries[0] - 1
		self.absoluteReturn = temp
		return temp
#Calculate the annualized return
	def annualizedReturn(self, dayConvention=365):
		try:
			temp_datediff = self.datediff(self.seriesDates[0],self.seriesDates[-1])
			temp_datediff = temp_datediff / dayConvention
		except AttributeError:
			raise AttributeError('You must define seriesDates attribute to use this subclass')
		temp = np.exp(np.sum(np.log(self.dailyReturns+1))) #Absolute Return + 1
		temp = np.log(temp)/temp_datediff
		self.annualizedReturn = temp
		return temp
#Calculate the sharpe ratio
	def sharpe(self, dayConvention=365, period=252):
		try:
			temp = self.annualizedReturn/self.volatility
		except TypeError:
			try:
				self.volatility(period)
				self.annualizedReturn(dayConvention)
				temp = self.annualizedReturn/self.volatility
			except TypeError:
				raise Warning('Could not calculate .annualizedReturn or .volatility')
		self.sharpe = temp
		return temp


######################################################################################################
######################################################################################################
############################# CODE - DISTRIBUTION SUBCLASS ###########################################
######################################################################################################
######################################################################################################

#Child Class of financialTimeSeries used for distribution analysis
#entryType = 'dailyReturns'
class distribution(financialTimeSeries):
#Initilization of the class and restrictions
	def __init__(self, timeSeries):
		#Initialize the father class
		financialTimeSeries.__init__(self, timeSeries, entryType='dailyReturns')
#Calculate 1st moment = average return
	def averageReturn(self):
		temp = np.mean(self.dailyReturns)
		self.averageReturn = temp
		return temp
#Calculate 2nd moment = variance
	def variance(self):
		temp = np.var(self.dailyReturns)
		self.variance = temp
		return temp
#Calculate 3rd moment = skewness
	def skewness(self):
		temp = skew(self.dailyReturns)
		self.skewness = temp
		return temp
#Calculate 4th moment = kurtosis
	def kurtosis(self):
		temp = kurtosis(self.dailyReturns)
		self.kurtosis = temp
		return temp
#Calculate VaR
	def VaR(self, period=5, interval=0.95):
		s = np.divide(self.timeSeries[period:], self.timeSeries[:-period]) - 1
		s = np.sort(s)
		j_0 = floor(self.NElements/(1/(1-interval)))
		VaR = s[j_0] + (self.NElements/(1/(1-interval)) - j_0) * (s[j_0] - s[j_0])
		self.VaR = VaR
		return VaR
#Calculate CVaR
	def CVaR(self, period=5, interval=0.95):
		#Generate period-days returns, then sort and calculate
		s = np.divide(self.timeSeries[period:], self.timeSeries[:-period]) - 1
		s = np.sort(s)
		j_0 = floor(self.NElements/(1/(1-interval)))
		if j_0 == 0:
			CVaR = s[0]
		elif j_0 == self.NElements/(1/(1-interval)):
			CVaR = (1/j_0) * np.sum(s[:j_0])
		else:
			CVaR = (1/(j_0-1)) * np.sum(s[:j_0-1])
		self.VaR = CVaR
		return CVaR


######################################################################################################
######################################################################################################
############################### CODE - DRAWDOWN SUBCLASS #############################################
######################################################################################################
######################################################################################################

#Child Class of financialTimeSeries used for drawdown analytics
#entryType = 'timeSeries'
class drawDown(financialTimeSeries):
#Initilization of the class and restrictions
	def __init__(self, timeSeries):
		#Initialize the father class
		financialTimeSeries.__init__(self, timeSeries, entryType='timeSeries')
		#Generate drawdowns attribute
		drawDownList = np.zeros(self.NElements)
		for i in range(1,self.NElements):
			drawDownList[i] = self.timeSeries[i]/np.max(self.timeSeries[:i+1]) - 1
		self.drawDownList = drawDownList
#Calculate MDD 
	def max(self):
		maxDrawDown = np.min(self.drawDownList)
		self.max = maxDrawDown
		return maxDrawDown
#Calculate MDD Index
	def maxIndex(self):
		maxDrawDownIndex = np.argmin(self.drawDownList)
		self.maxIndex = maxDrawDownIndex
		return maxDrawDownIndex
#Calculate the time to recovery
	def maxTimeToRecovery(self, periodType="w"):
		periodLength = self.periodTypeMatcherToBD(periodType)
		#Find Max Draw Down Index
		maxDrawDownIndex = np.argmin(self.drawDownList)
		#If no drawdown return 0
		if maxDrawDownIndex == 0:
			timeToRecovery = -1
			self.timeToRecovery = timeToRecovery
			return timeToRecovery
		#Find Max Value before MDD
		maxBeforeMDDIndex = np.argmax(self.timeSeries[:maxDrawDownIndex])
		maxBeforeMDD = np.max(self.timeSeries[:maxDrawDownIndex])
		#Find when recovery happened or returns -1
		recoveryIndex = -1
		for i in range(maxDrawDownIndex,self.NElements):
			if self.timeSeries[i] >= maxBeforeMDD:
				recoveryIndex = i
				break
		if recoveryIndex != -1:
			maxTimeToRecovery = max(0, (recoveryIndex - maxBeforeMDDIndex)/periodLength)
		else:
			maxTimeToRecovery = -1
		self.maxTimeToRecovery = maxTimeToRecovery
		self.maxDrawDownRecoveryIndex = recoveryIndex
		return maxTimeToRecovery
#Calculate index of TTR
	def maxTimeToRecoveryIndex(self):
		#Find Max Draw Down Index
		maxDrawDownIndex = np.argmin(self.drawDownList)
		#If no drawdown return 0
		if maxDrawDownIndex == 0:
			timeToRecovery = -1
			self.maxTimeToRecoveryIndex = timeToRecovery
			return maxTimeToRecoveryIndex
		#Find Max Value before MDD
		maxBeforeMDDIndex = np.argmax(self.timeSeries[:maxDrawDownIndex])
		maxBeforeMDD = np.max(self.timeSeries[:maxDrawDownIndex])
		#Find when recovery happened or returns -1
		maxTimeToRecoveryIndex = -1
		for i in range(maxDrawDownIndex,self.NElements):
			if self.timeSeries[i] >= maxBeforeMDD:
				maxTimeToRecoveryIndex = i
				break
		self.maxTimeToRecoveryIndex = maxTimeToRecoveryIndex
		return maxTimeToRecoveryIndex
#Calculate MDDs down to limit % of max DD  
	def table(self, limit=0.30):
		drawDownTable = self.drawDownList
		mdd = min(self.drawDownList)
		dd_list = list()
		max_list = list()
		time_list = list()
		recovery_list = list()
		while np.min(drawDownTable) < limit*mdd:
			new_dd = np.min(drawDownTable)
			new_time = np.argmin(drawDownTable)
			new_max = np.argmax(self.timeSeries[:new_time])
			new_maxValue = self.timeSeries[new_max]
			if new_time == 0:
				break
			else:
				new_recovery = -1
				for i in range(new_time, self.NElements):
					if self.timeSeries[i] >= new_maxValue:
						new_recovery = i
						break
			dd_list.append(new_dd)
			max_list.append(new_max)
			time_list.append(new_time)
			recovery_list.append(new_recovery)
			#Determine what to delete from DrawDownTable
			if new_recovery == -1: 
				to_delete_end = self.NElements
			else:
				to_delete_end = new_recovery
			for i in range(new_max, to_delete_end):
				drawDownTable[i] = 0
	#Create pandas dataframe
		drawDownTable = {
			'drawDown':dd_list, 
			'maxIndex': max_list, 
			'ddIndex': time_list, 
			'recoveryIndex': recovery_list
		}
		drawDownTable = pd.DataFrame(drawDownTable)
		self.table = drawDownTable
		return drawDownTable
#Get dates in table
	def datesTable(self, dates, limit=0.30):
		try:
			N = len(self.table)
		except TypeError:
			self.table()
			N = len(self.table)
		dates_dict = dict(zip(range(len(dates)),dates))
		self.table['maxIndex'] = self.table['maxIndex'].map(dates_dict)
		self.table['ddIndex'] = self.table['ddIndex'].map(dates_dict)
		self.table['recoveryIndex'] = self.table['recoveryIndex'].map(dates_dict)
		return self.table


######################################################################################################
######################################################################################################
################################ CODE - DRAWUP SUBCLASS ##############################################
######################################################################################################
######################################################################################################

#Child Class of financialTimeSeries used for drawUp analytics
#entryType = 'timeSeries'
class drawUp(financialTimeSeries):
#Initilization of the class and restrictions
	def __init__(self, timeSeries):
		#Initialize the father class
		financialTimeSeries.__init__(self, timeSeries, entryType='timeSeries')
		#Generate drawdowns attribute
		drawUpList = np.zeros(self.NElements)
		for i in range(1,self.NElements):
			drawUpList[i] = self.timeSeries[i]/np.min(self.timeSeries[:i+1]) - 1
		self.drawUpList = drawUpList
#Calculate MDD 
	def max(self):
		maxDrawUp = np.max(self.drawUpList)
		self.maxDrawUp = maxDrawUp
		return max
#Calculate MDD Index
	def maxIndex(self):
		maxDrawUpIndex = np.argmax(self.drawUpList)
		self.maxIndex = maxDrawUpIndex
		return maxDrawUpIndex
#Calculate MDDs down to limit % of max DD  
	def table(self, limit=0.30):
		drawUpTable = self.drawUpList
		mdu = max(self.drawUpList)
		du_list = list()
		min_list = list()
		time_list = list()
		refall_list = list()
		while np.max(drawUpTable) > limit*mdu:
			new_du = np.max(drawUpTable)
			new_time = np.argmax(drawUpTable)
			new_min = np.argmin(self.timeSeries[:new_time])
			new_minValue = self.timeSeries[new_min]
			if new_time == 0:
				break
			else:
				new_refall = -1
				for i in range(new_time, self.NElements):
					if self.timeSeries[i] <= new_minValue:
						new_refall = i
						break
			du_list.append(new_du)
			min_list.append(new_min)
			time_list.append(new_time)
			refall_list.append(new_refall)
			#Determine what to delete from DrawDownTable
			if new_refall == -1: 
				to_delete_end = self.NElements
			else:
				to_delete_end = new_refall
			for i in range(new_min, to_delete_end):
				drawUpTable[i] = 0
		#Create pandas dataframe
		drawUpTable = {
			'drawUp':du_list, 
			'maxIndex': min_list, 
			'ddIndex': time_list, 
			'refallIndex': refall_list
		}
		drawUpTable = pd.DataFrame(drawUpTable)
		self.table = drawUpTable
		return drawUpTable
#Get dates in table
	def datesTable(self, dates, limit=0.3):
		try:
			N = len(self.table)
		except TypeError:
			self.table()
			N = len(self.table)
		dates_dict = dict(zip(range(len(dates)),dates))
		self.table['maxIndex'] = self.table['maxIndex'].map(dates_dict)
		self.table['ddIndex'] = self.table['ddIndex'].map(dates_dict)
		self.table['refallIndex'] = self.table['refallIndex'].map(dates_dict)
		return self.table


######################################################################################################
######################################################################################################
################################# CODE - TTNH SUBCLASS ###############################################
######################################################################################################
######################################################################################################

#Child Class of financialTimeSeries used for time to new high analytics
#entryType='timeSeries'
class timeToNewHigh(financialTimeSeries):
#Initilization of the class and restrictions
	def __init__(self, timeSeries):
		#Initialize the father class
		financialTimeSeries.__init__(self, timeSeries, entryType='timeSeries')
#Calculate the time to new high
	def max(self, periodType="w"):
		periodLength = self.periodTypeMatcherToBD(periodType)
		timeDown = np.zeros(self.NElements)
		downToTheEnd = list()
		#Find when New High Was Reached
		for i in range(self.NElements):
			for j in range(i,self.NElements):
				timeDown[i] = j - i + 1
				if i != j and self.timeSeries[j] >= self.timeSeries[i]:
					break
		#If goes to the end, excludes to find real one
		#Except if end is the moment New High was reached
			if (timeDown[i] == self.NElements-i 
					and self.timeSeries[i] < self.timeSeries[self.NElements-1]):
				downToTheEnd.append(int(self.NElements-i))
				timeDown[i] = 0
		potentialTTNH = np.max(timeDown)
		if len(downToTheEnd) > 0:
			potentialNoTTNH = max(downToTheEnd)
		else:
			potentialNoTTNH = 0
		#If real one shorter, then no TTNH exists
		if potentialTTNH < potentialNoTTNH:
			maxTimeToNewHigh = -1
		else:
			maxTimeToNewHigh = (potentialTTNH-1)/periodLength
		self.max = maxTimeToNewHigh
		return maxTimeToNewHigh
#Calculate the time to new high start index
	def maxStartIndex(self, periodType="w"):
		periodLength = self.periodTypeMatcherToBD(periodType)
		timeDown = np.zeros(self.NElements)
		downToTheEnd = list()
		downToTheEnd_index = list()
		#Find when New High Was Reached
		for i in range(self.NElements):
			for j in range(i,self.NElements):
				timeDown[i] = j - i + 1
				if i != j and self.timeSeries[j] >= self.timeSeries[i]:
					break
		#If goes to the end, excludes to find real one
		#Except if end is the moment New High was reached
			if (timeDown[i] == self.NElements-i 
				and self.timeSeries[i] < self.timeSeries[self.NElements-1]):
				downToTheEnd.append(int(self.NElements-i))
				downToTheEnd_index.append(i)
				timeDown[i] = 0
		potentialTTNH = np.max(timeDown)
		if len(downToTheEnd) > 0:
			potentialNoTTNH = max(downToTheEnd)
		else:
			potentialNoTTNH = 0
		#If real one shorter, then no TTNH exists
		#The start index is the index of biggest timeDown that was moved to downToEnd
		if potentialTTNH < potentialNoTTNH:
			maxTimeToNewHighStartIndex = downToTheEnd_index[np.argmax(downToTheEnd)]
		else:
			maxTimeToNewHighStartIndex = np.argmax(timeDown)
		self.maxStartIndex = maxTimeToNewHighStartIndex
		return maxTimeToNewHighStartIndex
#Calculate the time to new high end index
	def maxEndIndex(self, periodType="w"):
		periodLength = self.periodTypeMatcherToBD(periodType)
		timeDown = np.zeros(self.NElements)
		downToTheEnd = list()
		downToTheEnd_index = list()
		#Find when New High Was Reached
		for i in range(self.NElements):
			for j in range(i,self.NElements):
				timeDown[i] = j - i + 1
				if i != j and self.timeSeries[j] >= self.timeSeries[i]:
					break
		#If goes to the end, excludes to find real one
		#Except if end is the moment New High was reached
			if (timeDown[i] == self.NElements-i 
					and self.timeSeries[i] < self.timeSeries[self.NElements-1]):
				downToTheEnd.append(int(self.NElements-i))
				downToTheEnd_index.append(i)
				timeDown[i] = 0
		potentialTTNH = np.max(timeDown)
		if len(downToTheEnd) > 0:
			potentialNoTTNH = max(downToTheEnd)
		else:
			potentialNoTTNH = 0
		#If real one shorter, then no TTNH exists
		#The end index is the last index
		if potentialTTNH < potentialNoTTNH:
			maxTimeToNewHighEndIndex = self.NElements-1
		else:
			maxTimeToNewHighEndIndex = np.argmax(timeDown) + max(timeDown)
		self.maxEndIndex = maxTimeToNewHighEndIndex
		return maxTimeToNewHighEndIndex
#Generate table with all TTNH
	def table(self, limit=0.30):
		#Initialize lists and variables
		ttnh_list = list()
		ttnh_start = list()
		ttnh_end = list()
		ttnh_mdd = list()
		to_check_start = [0]
		to_check_end = [self.NElements]
		checked_counter = 0
		checkedArea_sum = 0
		#Will loop on every subdivision it is given to look for TTNH inside
		while len(to_check_start) > checked_counter:
			start_index = int(to_check_start[checked_counter])
			end_index = int(to_check_end[checked_counter])
			#Check that reduction of size did not go down to zero
			if end_index-start_index == 0:
				checked_counter += 1
				checkedArea_sum += 1
			#If not, creates a subTimeSeries class object to find its TTNH
			else:
				subTimeSeries=np.zeros(end_index-start_index)
				subTimeSeries[0:end_index-start_index+1] = self.timeSeries[start_index:end_index]
				subObject = financialTimeSeries(subTimeSeries, 'timeSeries')
				subObject.timeToNewHigh()
				ttnh = int(subObject.timeToNewHigh.max("d"))
				start = int(subObject.timeToNewHigh.maxStartIndex("d") + start_index)
				end = int(subObject.timeToNewHigh.maxEndIndex("d") + start_index)
				#Rejects under 3 as they don't make any fucking sense
				if ((end - start > 3)
					and (ttnh != -1 or end == self.NElements -1)):
					ttnh_list.append(ttnh)
					ttnh_start.append(start)
					ttnh_end.append(end)
					#Calculate MDD on newly found area
					subTimeSeries2=np.zeros(end-start)
					subTimeSeries2[0:end-start+1] = self.timeSeries[start:end]
					subObject2 = financialTimeSeries(subTimeSeries2, 'timeSeries')
					subObject2.drawDown()
					mdd = subObject2.drawDown.max()
					ttnh_mdd.append(mdd)
					#Create the index of the up to two new area to explore
					if ((start == start_index and end == end_index) 
						or start == end):
						pass
					elif start == start_index:
						to_check_start.append(int(end))
						to_check_end.append(int(end_index))
					elif end == end_index:
						to_check_start.append(int(start_index))
						to_check_end.append(int(start))
					else:
						#Big debuging block - KEEP IT 
						#pr_int to fix when using, put like this so out of CTR+F
						"""
						pr_int(
							"Cas 4\n"
							+"__________________________________________\n" 
							+"INPUT\n"
							+"start: "+str(start_index)+" end: "+str(end_index)+"\n"
							+"------------------------------------------\n"
							+"To check 1\n"
							+"start: " + str(int(start_index)) + " end: " + str(int(start))+"\n"
							+"OUTPUT\n"
							+"start: " + str(int(start)) + " end: " + str (int(end))+"\n"
							+"To check 2\n"
							+"start: " + str(int(end)) + " end: " + str(int(end_index))+"\n"
							+"__________________________________________\n"
						)
						"""
						to_check_start.append(int(start_index))
						to_check_end.append(int(start))
						
						to_check_start.append(int(end))
						to_check_end.append(int(end_index))
			#The area was checked and divided
			checked_counter += 1
			checkedArea_sum += end - start
		#Remove TTNH under limit
		max_ttnh = max(ttnh_list)
		ttnh_index_to_keep = list()
		for i in range(len(ttnh_list)):
			if ttnh_list[i] > max_ttnh*limit:
				ttnh_index_to_keep.append(i)
		ttnh_list = [ttnh_list[i] for i in ttnh_index_to_keep]
		ttnh_start = [ttnh_start[i] for i in ttnh_index_to_keep]
		ttnh_end = [ttnh_end[i] for i in ttnh_index_to_keep]
		ttnh_mdd = [ttnh_mdd[i] for i in ttnh_index_to_keep]
		#Create pandas dataframe
		timeToNewHighTable = {
			'timeToNewHigh':ttnh_list, 
			'startIndex': ttnh_start, 
			'endIndex': ttnh_end, 
			'maxDrawDown': ttnh_mdd
		}
		timeToNewHighTable = pd.DataFrame(timeToNewHighTable).sort_values(
			by=['timeToNewHigh'],ascending=False)
		self.table = timeToNewHighTable
		return timeToNewHighTable
#Get dates in table
	def datesTable(self, dates, limit=0.3):
		try:
			N = len(self.table)
		except TypeError:
			self.table()
			N = len(self.table)
		dates_dict = dict(zip(range(len(dates)),dates))
		self.table['startIndex'] = self.table['startIndex'].map(dates_dict)
		self.table['endIndex'] = self.table['endIndex'].map(dates_dict)
		return self.table


######################################################################################################
######################################################################################################
############################## CODE - SUPPORLINE SUBCLASS ############################################
######################################################################################################
######################################################################################################

#Child Class of financialTimeSeries used for supportLine related methods
#entryType='timeSeries'
class supportLine(financialTimeSeries):
#Initilization of the class and restrictions
	def __init__(self, timeSeries, seriesDates):
		#Initialize the father class
		financialTimeSeries.__init__(self, timeSeries, entryType='timeSeries',
			entryDates=seriesDates)
		#Generate basis for top and bottom
		self.botStart = np.argmin(self.timeSeries)
		self.topStart = np.argmax(self.timeSeries)
		#Generate direction
		self.direction = np.sign(self.timeSeries[-1]/self.timeSeries[0]-1)
#Function for solvers:Fit line with rate	
	def lineCalculator(self, startPoint, rate, 
		startValue=10**99):
		newLine = np.zeros(self.NElements)
		newLine[:] = self.timeSeries
		if startValue != 10**99:
			for i in range(self.NElements):
				date_diff = self.datediff(self.seriesDates[startPoint],self.seriesDates[i])
				newLine[i] = startValue*(1+rate)**(date_diff/365)
		else:
			for i in range(self.NElements):
				date_diff = self.datediff(self.seriesDates[startPoint],self.seriesDates[i])
				newLine[i]= self.timeSeries[startPoint]*(1+rate)**(date_diff/365)
		return newLine
	def boundLineSolver(self, bound="bot", precision=0.0000001, tolerance=0.01):	
		#define variables	
		minRate = min(0,self.direction*0.99)
		maxRate = max(0,self.direction)
		newLine = np.zeros(self.NElements)
		#Generate the start position according to bound
		if bound == "top":
			startPoint = self.topStart
		else:
			startPoint = self.botStart
		#Loop to find rate
		while True:
			newRate = (minRate + maxRate)/2
			newLine = self.lineCalculator(startPoint, newRate)
			#Generate the position comparator according to bound
			if bound == "top":
				newPos = np.sum((self.timeSeries - newLine) <= 0) >= self.NElements*(1-tolerance)
			else:
				newPos = np.sum((self.timeSeries - newLine) >= 0) >= self.NElements*(1-tolerance)
			#Check that the newLine remains under/above the asset line
			if newPos:
				if self.direction == 1:
					diffRate = abs(minRate - newRate)
				else:
					diffRate = abs(maxRate - newRate)
				#If true: precision desired was achieved
				if diffRate < precision:
					break
				#If not: keep going
				else:
					if self.direction == 1:
						minRate = newRate
					else:
						maxRate = newRate
			#Got above/under the curve so need to go lower/higher
			else:
				if self.direction == 1:
					maxRate = newRate
				else:
					minRate = newRate
		#Generate attribute of bound
		if bound == "top":
			self.topLine = newLine
			self.topRate = newRate
			return newRate
		else:
			self.botLine = newLine
			self.botRate = newRate
			return newRate
#Check that the lines fitted are not completely absurd
	def sane(self, tolerance=1):
		if ((np.max(self.topLine - self.botLine)) > 
			tolerance*(max(self.timeSeries) - min(self.timeSeries))):
			return False
		else:
			return True
#Generate a mid lane between top an bot
	def midLineCalculator(self):
		newLine = (self.botLine 
				+ (1/2) *(self.topLine-self.botLine))
		self.midLine = newLine
		return newLine
#Generate additionalLine between top and bot
	def additionalLineCalculator(self, additionalLine=3):
		additionalLineList = list()
		for i in range(additionalLine):
			newLine = (self.botLine 
				+ (i +1)/(additionalLine + 1)*(self.topLine-self.botLine))
			additionalLineList.append(newLine)
		self.additionalLineList = additionalLineList
		self.additionalLineType = 'divided'
		return additionalLineList
#Solve additionalLine between top and bot
	def additionalLineSolver(self, additionalLine=3, tolerance=0.01, maxIteration=100):
		additionalLineList = list()
		additionalLineRate = list()
		#NElement is not accurate as some terms are above topLine / under botLine
		#and so should not be counted in our division of the area
		toCountForUnder = np.sum(
			np.logical_and(
				self.botLine < self.timeSeries, 
				self.timeSeries <= self.topLine)
			)
		#Need to find on which side the two support line converge
		#this is important to know what startpoint to use
		diffBeg = self.topLine[0] - self.botLine[0]
		diffEnd = self.topLine[self.NElements-1] - self.botLine[self.NElements-1]
		if diffEnd > diffBeg:
			startpoint = 0
			timeSeriesToUse = self.botLine
			minRateToUse = self.botRate
			maxRateToUse = 1
			toCheckAfter = "top"
			startValueList = [self.botLine[0] + diffBeg*(i+1)/(
				additionalLine+1) for i in range(
				additionalLine)]
		else:
			startpoint = self.NElements-1
			timeSeriesToUse = self.topLine
			minRateToUse = -1
			maxRateToUse = self.topRate
			toCheckAfter ="bot"
			startValueList = [self.botLine[-1] + diffEnd*(i+1)/(
				additionalLine+1) for i in range(
				additionalLine)]
		#Iterate for every line to solve
		for i in range(additionalLine):
			startValue = startValueList[i]
			#Count how many elements need to be under line to be solved
			toCount = toCountForUnder * (i+1)/(additionalLine+1)
			#Now takes tolerance into account and create an acceptable interval
			toCountMin = int(floor(toCount * (1-tolerance)))
			toCountMax = int(floor(toCount * (1+tolerance)))
			#Determine lower and upper rates to do dichotomy between
			#For first iteration use minRateToUse as lower rate
			if i == 0:
				minRate = minRateToUse
			#Then use rate found for line under as lower rate
			else: 
				minRate = additionalLineRate[i-1]
			#As bot to top proces, always use maxRateToUse as upper rate
			minRate = minRateToUse
			maxRate = maxRateToUse
			#Iteration limit
			iterationCounter = 0
			while True:
				iterationCounter += 1
				if iterationCounter == maxIteration:
					break
				newRate = (minRate + maxRate)/2
				newLine = self.lineCalculator(startpoint, newRate,startValue=startValue)
				checker = np.sum(self.timeSeries <= newLine)
				#1. Not enough data point under
				if checker < toCountMin:
					#Then rate must go up to bring more data point under
					minRate = newRate
				#2. Enough data point between
				elif checker <= toCountMax:
					#Then we have enough data points in between, line is good
					break
				#3. Too many data point under
				else:
					#Then rate must go down to have less data point under
					maxRate = newRate
			#Return results either from solving or too many iterations
			additionalLineList.append(newLine)
			additionalLineRate.append(newRate)
		#Changes Bot or Top if necessary:
		if toCheckAfter == "bot" and additionalLineList[0][0] < self.botLine[0]:
			self.botLine = additionalLineList[0] - (self.topLine[-1] - self.botLine[-1])
		elif toCheckAfter == "top" and  additionalLineList[-1][-1] > self.topLine[-1]:
			self.topLine = additionalLineList[-1] + (self.topLine[0] - self.botLine[0])
		else:
			pass
		#Check that solver managed
		checker = [abs(additionalLineRate[i+1]-additionalLineRate[i]) for i in range(len(
			additionalLineRate)-1)]
		print(checker)
		if max(checker) < 0.0001:
			self.solverFailed = True
		else:
			self.solverFailed = False
		self.additionalLineList = additionalLineList
		self.additionalRateList = additionalLineRate
		self.additionalLineType = 'solved'
		return additionalLineList
#Function to determine between which lines a single point is located
	def positionOfPointBetweenLines(self, pointIndex):
		pointValue = self.timeSeries[pointIndex]
		counter = 0
		#Load bounLines and generate Sky
		customLineList = list()
		customLineList[:] = self.additionalLineList[:]
		customLineList.insert(0,self.botLine)
		customLineList.append(self.topLine)
		skyLine   = np.ones(np.shape(customLineList[0])[0]) *  10**99
		customLineList.append(skyLine)
		#Check under wich line it is, break when found
		for line in customLineList:
			lineValue = line[pointIndex]
			if pointValue <= lineValue:
				break
			else:
				counter += 1
		return counter
#Return Bool arrays to know if values are between two lines
	def positionBetweenLines(self, additionalLine=3, tolerance=0.01, 
		addLineType='divided', toleranceSolver=0.01, maxIterationSolver=100):
		#Generate all attributes
		if not hasattr(self, 'botLine'):
			self.boundLineSolver(bound="bot",tolerance=tolerance)
		if not hasattr(self, 'topLine'):
			self.boundLineSolver(bound="top",tolerance=tolerance)
		if addLineType == 'divided':
			if not hasattr(self, 'midLine'):
				self.midLineCalculator()
		#Here re-generate lines according to addLineType
		if addLineType == 'divided':
			self.additionalLineCalculator(additionalLine=additionalLine)
		elif addLineType == 'solved':
			self.additionalLineSolver(additionalLine=additionalLine,
				tolerance=toleranceSolver, maxIteration=maxIterationSolver)
		else:
			raise ValueError("addLineType must be 'divided' or 'solved'")
		#Will use bool arrays to determine what lines points are between
		allLines = list()
		allLines[:] = self.additionalLineList[:]
		allLines.insert(0,self.botLine)
		allLines.append(self.topLine)
		valuesAreAboveList = list()
		#All is supperior to the ground so need to always be true
		ValuesAreAbove = np.ones(self.NElements, dtype=bool)
		valuesAreAboveList.append(ValuesAreAbove)
		#check what is inferior/supperior to existing lines
		for observedLine in allLines:
			ValuesAreAbove = self.timeSeries > observedLine
			valuesAreAboveList.append(ValuesAreAbove)
		#All is inferior to the sky so need to always be false
		ValuesAreAbove = np.zeros(self.NElements, dtype=bool)
		valuesAreAboveList.append(ValuesAreAbove)
		#Now substract above_k and above_k+1 to get between_k&k+1
		valuesAreInBetween = np.zeros(self.NElements, dtype=bool)
		valuesAreInBetweenList = list()
		for i in range(len(allLines) + 1):
			#v[i]=true and v[i+1]=false indicates that it is above v[i] but under v[i+1]
			#so we invert the true and false in v[i+1] to be able to look for pairs of true
			valuesAreInBetween = np.logical_and( valuesAreAboveList[i],
				np.logical_not(valuesAreAboveList[i+1]) )
			valuesAreInBetweenList.append(valuesAreInBetween)
		self.betweenLines = valuesAreInBetweenList
		return valuesAreInBetweenList
#Generate the table with avgReturn, regression and nextPosition
	def table(self, additionalLine=3, period=21, tolerance=0.01, 
		addLineType='divided', toleranceSolver=0.01, maxIterationSolver=100):
		#Generate all attributes
		betweenList = self.positionBetweenLines(additionalLine=additionalLine, 
			tolerance=tolerance, 
			addLineType=addLineType, 
			toleranceSolver=toleranceSolver, 
			maxIterationSolver=maxIterationSolver)
		averageReturnAfter = list()
		totalItem = list()
		posItem = list()
		negItem = list()
		#Determine performance from each inbetween position
		subStart = self.timeSeries[:-period]
		subEnd = self.timeSeries[period:]
		#Return position for each endPosition, 
		#taking into account that indexes were reduced by period
		subNewPos = np.zeros(self.NElements - period)
		for i in range(self.NElements - period):
			subNewPos[i] = self.positionOfPointBetweenLines(period + i)
		#Generate the meta list and list for k-pos matrix
		kPosMatrix = list()
		for i in range(additionalLine + 3):
			kPosMatrix.append(list())
		counterKposMatrix = 0
		#Does the filtering on each subLine
		for subList in betweenList:
			filterStart = subList[:-period]
			startValue = subStart[filterStart]
			endValue = subEnd[filterStart]
			newPosValue = subNewPos[filterStart]
			if len(startValue) > 0:
				#Save results in lists
				averageReturnAfter.append(np.mean(np.divide(endValue, startValue)-1))
				totalItem.append(np.shape(endValue)[0])
				posItem.append(np.sum((np.divide(endValue, startValue)-1) > 0))
				negItem.append(np.sum((np.divide(endValue, startValue)-1) <= 0))
				#Generate the to k-pos matrix
				for i in range(additionalLine + 3):
					countInPos_i = np.sum(newPosValue == i)
					kPosMatrix[counterKposMatrix].append(countInPos_i)
			else:
				averageReturnAfter.append(0)
				totalItem.append(0)
				posItem.append(0)
				negItem.append(0)
				#Generate the to k-pos matrix with 0-s since no value
				for i in range(additionalLine + 3):
					kPosMatrix[counterKposMatrix].append(0)
			#After the if statements, increment counterKposMatrix
			counterKposMatrix += 1
		#Transpose k-Pos matrix so that it is "To" instead of "from"
		#If not transposed; will tell for each line where the strat was
		#If transposed: will tell for each line where the strategy went
		temp = kPosMatrix[0:]
		kPosMatrix = [[row[i] for row in temp] for i in range(len(temp[0]))]
		#Generate lines names and k-pos matrix column names for pd dataframe
		linesNames = list()
		kPosMatrixNames = list()
		for i in range(len(averageReturnAfter)):
			if i == 0:
				entry  = str("< botLine")
				entry2 = str("to < botLine")
			elif i == len(averageReturnAfter)-1:
				entry  = str("> topLine")
				entry2 = str("to > topLine")
			else:
				minPerc = int(floor((i-1)/(len(averageReturnAfter)-2)*100))
				maxPerc = int(floor((i  )/(len(averageReturnAfter)-2)*100))
				entry   = str(str(minPerc) + "% to " + str(maxPerc) + "%")
				entry2  = str("to " + str(minPerc) + "% - " + str(maxPerc) + "%")
			linesNames.append(entry)	
			kPosMatrixNames.append(entry2)
		#Turn meta lists / matrices into dictionnaries
		allRegDict = {
			'line' : linesNames, 
			'averageReturn' : averageReturnAfter, 
			'#Cases' : totalItem,
			'#PosCases' : posItem,
			'#NegCases' : negItem
			}
		kPosMatrixDict = dict(zip(kPosMatrixNames, kPosMatrix))
		allInfo = dict(allRegDict)
		allInfo.update(kPosMatrixDict)
		#Generate pd dataframe
		table = pd.DataFrame(allInfo)
		self.table_attr = table
		self.table_type = addLineType
		return table		
#Generate current proba from table of support line
	def currentProba(self, additionalLine=3, period=21, tolerance=0.01,
		addLineType='divided', toleranceSolver=0.01, maxIterationSolver=100):
		#regenerate table if not as we want it or generate it if do not exist
		try:
			if self.table_type != addLineType:
				self.table(
					additionalLine=additionalLine, 
					period=period, 
					tolerance=tolerance,
					addLineType=addLineType,
					toleranceSolver=toleranceSolver,
					maxIterationSolver=maxIterationSolver
					)
		except AttributeError:
			self.table(
				additionalLine=additionalLine, 
				period=period, 
				tolerance=tolerance,
				addLineType=addLineType,
				toleranceSolver=toleranceSolver,
				maxIterationSolver=maxIterationSolver
				)
		#find desired info in table
		currentPosBetweenLines = self.positionOfPointBetweenLines(self.NElements-1)
		desiredCol=self.table_attr.iloc[currentPosBetweenLines,:]
		probaStateList = list()
		probaNameList = list()
		for i in range(additionalLine+3):
			probaState = desiredCol.iloc[5+i]/desiredCol.iloc[2]
			probaStateList.append(probaState)
			probaNameList.append(str(i))
		probaStateList.insert(0, desiredCol.iloc[4]/desiredCol.iloc[2])
		probaStateList.insert(1, desiredCol.iloc[3]/desiredCol.iloc[2])
		probaNameList.insert(0, "down")
		probaNameList.insert(1, "up")
		probaTable = dict(zip(probaNameList, probaStateList))
		column = str(addLineType+" - "+"Pos="+str(currentPosBetweenLines))
		currentProba = pd.DataFrame(probaStateList, index=probaNameList, columns=[column])
		self.currentProba_attr = currentProba
		return currentProba	
#Plot the whole thing
	def plot(self, additionalLine=3, tolerance=0.01, addLineType='divided'):
		#Get all attributes
		if not hasattr(self, 'botLine'):
			self.boundLineSolver(bound="bot", tolerance=tolerance)
		if not hasattr(self, 'topLine'):
			self.boundLineSolver(bound="top", tolerance=tolerance)
		if addLineType == 'divided':
			if not hasattr(self, 'midLine'):
				self.midLineCalculator()
			if not hasattr(self, 'additionalLine') or self.addLineType != 'divided':
				self.additionalLineCalculator(additionalLine=additionalLine)
		elif addLineType == 'solved':
			if not hasattr(self, 'additionalLine') or self.addLineType != 'solved':
				self.additionalLineSolver(additionalLine=additionalLine, 
					tolerance=0.01, maxIteration=100)
		else:
			raise ValueError("additionalLine must be 'divided' or 'solved'")
		#Plot
		import matplotlib.pyplot as plt
		for i in range(len(self.additionalLineList)):
			plt.plot(self.seriesDates, self.additionalLineList[i], color='grey')
		#Check midline as it depends on addLineType
		if addLineType == 'divided':
			plt.plot(self.seriesDates, self.midLine, color='black')
		elif addLineType == 'solved':
			if len(self.additionalLineList)%2 == 1:
				midIndex = len(self.additionalLineList)//2
				plt.plot(self.seriesDates, self.additionalLineList[midIndex], color='black')
		#Plot what is common
		plt.plot(self.seriesDates, self.timeSeries)
		plt.plot(self.seriesDates, self.botLine, color='red')
		plt.plot(self.seriesDates, self.topLine, color='green')
		plt.title("Fit with tolerance="+str(tolerance)
			+" and additionalLine="+str(additionalLine))
		plt.show()
		return True


######################################################################################################
######################################################################################################
############################### CODE - TECHNICAL SUBCLASS ############################################
######################################################################################################
######################################################################################################

#Child Class of financialTimeSeries used for technical analysis
#entryType = 'timeSeries'
class technical(financialTimeSeries):
#Initilization of the class and restrictions
	def __init__(self, timeSeries):
		#Initialize the father class
		financialTimeSeries.__init__(self, timeSeries, entryType='timeSeries')
	def initializationRelativeStrengthIndex(self, period=14):
		dailyChange = np.zeros(self.NElements)
		dailyChange[1:]=self.timeSeries[1:]-self.timeSeries[:-1]
		subArray = dailyChange[-period+1:]
		avgDailyPos = np.sum( subArray[subArray> 0])/period
		avgDailyNeg = np.sum(-subArray[subArray<=0])/period
		RS = avgDailyPos/avgDailyNeg
		if avgDailyNeg == 0:
			RSI = 1
		else:
			RSI = 1 - 1/(1+RS)
		self.initializationRelativeStrengthIndex = RSI
		return RSI
	def relativeStrengthIndex(self, period=14):
		RSI = np.ones(self.NElements)*0.5
		dailyChange = np.zeros(self.NElements)
		dailyChange[1:]=self.timeSeries[1:]-self.timeSeries[:-1]
		subArray = dailyChange[0:period+1]
		avgDailyPos = np.sum( subArray[subArray> 0])/period
		avgDailyNeg = np.sum(-subArray[subArray<=0])/period
		RS = avgDailyPos/avgDailyNeg
		if avgDailyNeg == 0:
			RSI[period] = 1
		else:
			RSI[period] = 1 - 1/(1+RS)
		for i in range(period+1,self.NElements):
			avgDailyPos = (avgDailyPos * (period - 1) + max(0, dailyChange[i]))/period
			avgDailyNeg = (avgDailyNeg * (period - 1) + max(0,-dailyChange[i]))/period
			RS = avgDailyPos/avgDailyNeg
			if avgDailyNeg == 0:
				RSI[i] = 0
			else:
				RSI[i] = 1 - 1/(1+RS)
		self.relativeStrengthIndex = RSI
		return RSI
	def exponentialMovingAverage(self, 
		alternativeTimeSeries=np.zeros(1), period=14, smoothingFactor=2):
		adjustingFactor = smoothingFactor/(1+period)
		#Calculate EMA of timeSeries
		if np.shape(alternativeTimeSeries) == np.shape(np.zeros(1)):
			initialTerm = np.mean(self.timeSeries[period:])
			EMA = np.ones(self.NElements)*initialTerm
			for i in range(period, self.NElements):
				EMA[i] = (self.timeSeries[i] - EMA[i-1])*adjustingFactor + EMA[i-1]
		#Calculate EMA of any series fed to it
		else:
			initialTerm = np.mean(alternativeTimeSeries[period:])
			N = np.shape(alternativeTimeSeries)[0]
			EMA = np.ones(N)*initialTerm
			for i in range(period, N):
				EMA[i] = (alternativeTimeSeries[i] - EMA[i-1])*adjustingFactor + EMA[i-1]
		return EMA
	def movingAverageConvergenceDivergence(self, 
		period1=12, period2=26, smoothingFactor=2):
		MACD = np.zeros(self.NElements)
		maxPeriod = max(period1, period2)
		line1 = self.exponentialMovingAverage(period=period1)
		line2 = self.exponentialMovingAverage(period=period2)
		MACD[maxPeriod:] = line1[maxPeriod:] - line2[maxPeriod:]
		return MACD
	def macdSignals(self, 
		period1=12, period2=26, signalPeriod=9, smoothingFactor=2):
		MACD = self.movingAverageConvergenceDivergence(
			period1=period1, period2=period2, 
			smoothingFactor=smoothingFactor)
		signalLine = self.exponentialMovingAverage(
			alternativeTimeSeries=MACD, 
			period=signalPeriod, smoothingFactor=smoothingFactor)
		#By default: no signal
		macdSignals = np.zeros(self.NElements)
		maxPeriod=max(period1, period2)
		for i in range(maxPeriod,self.NElements):
			currentDiff = np.sign(MACD[i] - signalLine[i])
			lastDiff = np.sign(MACD[i-1] - signalLine[i-1])
			#First case: MACD went under signal line - sell
			if currentDiff < lastDiff:
				macdSignals[i] = -1
			#Second case: MACD went above signal line - buy
			elif currentDiff > lastDiff:
				macdSignals[i] =  1
		#By construction first is always signal: prevent it
		macdSignals[maxPeriod] = 0
		self.macd = MACD
		self.macdSignalLine = signalLine
		self.macdSignals = macdSignals
		return macdSignals
	def plot(self, dates, 
		periodRSI=14,
		period1MACD=12, period2MACD=26, signalPeriodMACD=9, smoothingFactorMACD=2):
		#Generate RSI
		self.relativeStrengthIndex(
			period=periodRSI)
		#Generate MACD
		self.macdSignals(
			period1=period1MACD, 
			period2=period2MACD, 
			signalPeriod=signalPeriodMACD, 
			smoothingFactor=smoothingFactorMACD)
		#Generate Graph
		import matplotlib.pyplot as plt
		maxPeriod=max(
			max(period1MACD,period2MACD)+signalPeriodMACD, 
			periodRSI)
		toPlot1 = self.macd
		toPlot2 = self.macdSignalLine
		toPlot3 = toPlot1 - toPlot2
		toPlot4 = self.macdSignals*0.5+0.5 #Reversed to correspond to RSI
		toPlot5 = 1-self.relativeStrengthIndex #Reversed cause fuck you
		toPlot6 = np.ones(self.NElements)*0.7
		toPlot7 = np.ones(self.NElements)*0.3
		fig,ax = plt.subplots()
		ax2=ax.twinx()
		ax2.bar(dates[maxPeriod:], toPlot3[maxPeriod:], color='grey')
		ax.plot(dates[maxPeriod:], toPlot1[maxPeriod:], color='#00915A')
		ax.plot(dates[maxPeriod:], toPlot2[maxPeriod:], color='#B65042')
		ax.legend(('MACD','Signal'),loc='upper right')
		ax2.legend(('Histogram'),loc='lower right')
		plt.show()
		plt.plot(dates[maxPeriod:], toPlot4[maxPeriod:], color='grey')
		plt.plot(dates[maxPeriod:], toPlot5[maxPeriod:], color='black')
		plt.plot(dates[maxPeriod:], toPlot6[maxPeriod:], color='#006B61')
		plt.plot(dates[maxPeriod:], toPlot7[maxPeriod:], color='#6B2920' )
		plt.legend(('MACD Signal','1-RSI','Buy','Sell'))
		plt.show()
		return True


######################################################################################################
######################################################################################################
############################### CODE - IMPORT FUNCTIONS ##############################################
######################################################################################################
######################################################################################################

### Import data functions for test purposes
## From Yahoo Finance

def importDataTimeSeries(ticker, start, end="today", dataType="Close"):
	data = yf.download(ticker, start, end)
	timeSeries = data[dataType].to_numpy()
	dates = np.atleast_1d(np.asarray(data.index.array))
	timeSeries = financialTimeSeries(timeSeries, entryType='timeSeries',
		entryDates=dates,
		subclassesToInitialize=np.ones(nSC))
	return timeSeries
## How To Use:
#Ticker="INFO"
#a = importDataTimeSeries(Ticker,start="2010-01-01", end="2020-06-01")

## From CSV file
def fileLoader(path, colSeries=1):
	import csv
	with open(path, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		price_history_to_clean = list(reader)
	#transpose lists
	temp = price_history_to_clean[0:]
	price_history_to_clean = [[row[i] for row in temp] for i in range(len(temp[0]))]
	#create class object
	timeSeries = np.array(price_history_to_clean[colSeries][1:], dtype=float)
	dates = np.array(price_history_to_clean[0][1:], dtype='datetime64[ns]')
	timeSeries = financialTimeSeries(timeSeries, entryType='timeSeries',
		entryDates=dates,
		subclassesToInitialize=np.ones(nSC))
	return timeSeries
## How to use:
#path = "C:\\Users\\33695\\Desktop\\QIS_Data\\aggregatedTimeSeries.csv"
#a = fileLoader(path, colSeries=2)


######################################################################################################
######################################################################################################
################################## JUNK AREA TO TEST #################################################
######################################################################################################
######################################################################################################


#Test technicals
"""
Ticker="INFO"

a = importDataTimeSeries(Ticker,start="2010-01-01", end="2020-06-01")

print(a.supportLine.table(period=21, tolerance=0.01, addLineType='divided'))
a.supportLine.plot()
a.technical.plot(a.seriesDates)

b = importDataTimeSeries(Ticker,start="2020-08-25", end="2020-09-25")
print(b.supportLine.currentProba(period=21, tolerance=0.01))

c = importDataTimeSeries(Ticker,start="2020-06-01", end="2020-09-24")
print(c.supportLine.currentProba(period=21, tolerance=0.01))

b.supportLine.plot()
b.technical.plot(b.seriesDates)

c.supportLine.plot()
c.technical.plot(c.seriesDates)
"""










