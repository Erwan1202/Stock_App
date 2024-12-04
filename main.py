import pandas as pd
from prophet import Prophet
import yfinance as yf
import datetime as dt

start_date = dt.datetime.now() - dt.timedelta(days=365)
end_date = dt.datetime.now()


def loads_data():
    data = yf.download('AAPL', start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']
    print(data.head())
    return data

loads_data()

