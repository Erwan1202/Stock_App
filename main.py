from datetime import date 

import yfinance as yf
from prophet import prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Load data
data = yf.download("AAPL", start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))

# Fit model
model = prophet.Prophet()
model.fit(data)


