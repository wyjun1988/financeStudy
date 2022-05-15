#stationary time series
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss 
import yfinance as yf
from matplotlib import pyplot as plt

def adf_test(x):
    indices = ['Test statistic', 'p-value', '# of lags used', '# of observations used']
    adf_test = adfuller(x, autolag='AIC')
    results = pd.Series(adf_test[0:4], index=indices)
    for key, value in adf_test[4].items():
        results[f'Critical Value ({key})'] = value
    
    return results

df = yf.download(tickers='AAPL', start='2011-01-01', end='2020-12-31', progress=False)
adf_test(df.Close)

def kpss_test(x, h0_type='c'):
    indices = ['Test statistic', 'p-value', '# of lags' ]
    kpss_test = kpss(x, regression=h0_type)
    results = pd.Series(kpss_test[0:3], index=indices)
    for key, value in kpss_test[3].items():
        results[f'Critical Value ({key})'] = value
    
    return results

kpss_test(df.Close)

N_LAGS= 40
SIGNIFICANCE_LEVEL = 0.05

fig, ax = plt.subplots(2,1)
plot_acf(df.Close, ax=ax[0], lags=N_LAGS, alpha= SIGNIFICANCE_LEVEL)
plot_pacf(df.Close, ax=ax[1], lags=N_LAGS, alpha= SIGNIFICANCE_LEVEL)
