import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import yfinance as yf

df_sp_500_price = yf.download(tickers='^GSPC', period='7y', interval='1d')
print(df_sp_500_price)

# Plotting S&P 500 stock prices
fig = go.Figure()

fig.add_trace(go.Candlestick(x=df_sp_500_price.index, open=df_sp_500_price['Open'], high=df_sp_500_price['High'], low=df_sp_500_price['Low'],
                             close=df_sp_500_price['Close'], name='Market data'))

fig.update_layout(title='S&P 500 live share price evolution', yaxis_title='Stock price (USD per shares)')

fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(
    buttons=list([
        #dict(count=15, label='15m', step='minute', stepmode='backward'),
        #dict(count=45, label='45m', step='minute', stepmode='backward'),
        #dict(count=1, label='HTD', step='hour', stepmode='todate'),
        #dict(count=1, label='1h', step='hour', stepmode='backward'),
        #dict(count=3, label='3h', step='hour', stepmode='backward'),
        #dict(count=6, label='6h', step='hour', stepmode='backward'),
        #dict(count=12, label='12h', step='hour', stepmode='backward'),
        #dict(count=24, label='24h', step='hour', stepmode='backward'),
        dict(count=7, label='7d', step='day', stepmode='backward'),
        dict(count=14, label='14d', step='day', stepmode='backward'),
        dict(count=30, label='30d', step='day', stepmode='backward'),
        dict(count=60, label='60d', step='day', stepmode='backward'),
        dict(count=90, label='90d', step='day', stepmode='backward'),
        dict(count=120, label='120d', step='day', stepmode='backward'),
        dict(count=365, label='1y', step='day', stepmode='backward'),
        dict(count=730, label='2y', step='day', stepmode='backward'),
        dict(count=1095, label='3y', step='day', stepmode='backward'),
        dict(count=1460, label='4y', step='day', stepmode='backward'),
        dict(count=1825, label='5y', step='day', stepmode='backward'),
        dict(count=2190, label='6y', step='day', stepmode='backward'),
        dict(count=2555, label='7y', step='day', stepmode='backward'),
        dict(step='all')
    ])
))

fig.show()

# We are only looking for the closing price for each record
df_sp_500_price_closing = pd.DataFrame(df_sp_500_price['Close']).rename(columns={'Close': 'Price'})
print(df_sp_500_price_closing.head())

df_sp_500_price_closing.plot(figsize=(20, 14))
plt.ylabel('S&P 500 Price')
plt.title('Price of S&P 500 in the last 7 years with 1 day intervals', fontsize=16)
plt.legend(fontsize=14)

# Fundtion to generate future dates for prediction data
def get_future_dates(start_date, into_future, offset=1):
    """
    Returns array of datetime values ranging from start_date to start_date +
    into_future (horizon).

    Parameters
    ----------
    start_date: Date to start range (np.datetime64).
    into_future: Number of days to add onto start date for range (int).
    offset: Number of days to offset start_date by (default = 1).
    """
    start_date = start_date + np.timedelta64(offset, "D")
    end_date = start_date + np.timedelta64(into_future, "D")

    return np.arange(start_date, end_date, dtype="datetime64[D]")