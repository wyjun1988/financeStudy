#compensated_stationary_time_series.py
import cpi
import pandas as pd
from datetime import date
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import yfinance as yf
import numpy as np
cpi.update()

DEFL_DATE = date(2011, 12, 31)

df = yf.download(tickers='AAPL', start='2011-01-01', end='2020-12-31', progress=False)
df['dt_index'] = df.index.map(lambda x: x.to_pydatetime().date())
df['price'] = df.Close
df['price_deflated'] = df.apply(lambda x: cpi.inflate(x.price, x.dt_index, DEFL_DATE), axis=1)

df[['price', 'price_deflated']].plot(title='AAPL price(deflated)');

WINDOW=12
selected_columns = ['price_log', 'rolling_mean_log', 'rolling_std_log']

df['price_log'] = np.log(df.price_deflated)
df['rolling_mean_log'] = df.price_log.rolling(window=WINDOW).mean()
df['rolling_std_log'] = df.price_log.rolling(window=WINDOW).std()

df[selected_columns].plot(title='AAPL price logged');

# differential
selected_columns = ['price_log_diff', 'roll_mean_log_diff', 'roll_std_log_diff']
df['price_log_diff'] = df.price_log.diff(1)
df['roll_mean_log_diff'] = df.price_log_diff.rolling(WINDOW).mean()
df['roll_std_log_diff'] = df.price_log_diff.rolling(WINDOW).std()

df[selected_columns].plot(title='AAPL price 1st diff')

def adf_test(x):
    indices = ['Test statistic', 'p-value', '# of lags used', '# of observations used']
    adf_test = adfuller(x, autolag='AIC')
    results = pd.Series(adf_test[0:4], index=indices)
    for key, value in adf_test[4].items():
        results[f'Critical Value ({key})'] = value
    
    return results

from pmdarima.arima import ndiffs, nsdiffs

print(f"Suggested # of differences (ADF): {ndiffs(df.price, test='adf')}")
print(f"Suggested # of differences (KPSS): {ndiffs(df.price, test='kpss')}")
print(f"Suggested # of differences (PP): {ndiffs(df.price, test='pp')}")