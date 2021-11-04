from yahoo_fin import stock_info as si
 
# default daily data
#daily_data = si.get_data("amzn") 
 
# get weekly data
#weekly_data = si.get_data("amzn", interval = "1wk")
 
# get monthly data
#monthly_data = si.get_data("amzn", interval = "1mo")

# get AAPL balance sheet
#a = si.get_balance_sheet("aapl")
 
# get AAPL cash flow info
#a = si.get_cash_flow("aapl") 
 
# get AAPL income statement
#a= si.get_income_statement("AAPL")

# get anaylsts previsions
#a= si.get_analysts_info("AAPL")


a = si.get_financials("AAPL")
print(type(a) )