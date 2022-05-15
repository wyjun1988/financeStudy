# exponential_smoothing.py
import matplotlib.pyplot as plt
import seaborn as sns

plt.set_cmap('cubehelix')
sns.set_palette('cubehelix')

COLORS = [plt.cm.cubehelix(x) for x in [0.1, 0.3, 0.5, 0.7]]

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from statsmodels.tsa.holtwinters import (ExponentialSmoothing, SimpleExpSmoothing, Holt)

df = yf.download('AAPL', start='2010-01-01', end='2021-12-31', adjusted=True, progress=False)

aapl = df.resample('M').last().rename(columns={'Adj Close': 'adj_close'}).adj_close

train_indices = aapl.index.year<2021
aapl_train = aapl[train_indices]
aapl_test = aapl[~train_indices]

test_length = len(aapl_test)

aapl.plot(title="AAPL stock price")

ses_1 = SimpleExpSmoothing(aapl_train).fit(smoothing_level=0.2)
ses_forecast_1 = ses_1.forecast(test_length)

ses_2 = SimpleExpSmoothing(aapl_train).fit(smoothing_level=0.5)
ses_forecast_2 = ses_2.forecast(test_length)

ses_3 = SimpleExpSmoothing(aapl_train).fit()
alpha = ses_3.model.params['smoothing_level']
ses_forecast_3 = ses_3.forecast(test_length)

aapl.plot(color=COLORS[0], title='Simple exponential smoothing', label = 'Actual', legend = True)

ses_forecast_1.plot(color=COLORS[1], legend = True, label=r'$\alpha=0.2$')
ses_1.fittedvalues.plot(color=[COLORS[1]])

ses_forecast_2.plot(color=COLORS[2], legend = True, label=r'$\alpha=0.5$')
ses_2.fittedvalues.plot(color=[COLORS[2]])

ses_forecast_3.plot(color=COLORS[3], legend = True, label=r'$\alpha={0:.4}$'.format(alpha))
ses_3.fittedvalues.plot(color=[COLORS[3]])

hs_1 = Holt(aapl_train).fit()
hs_forecast_1 = hs_1.forecast(test_length)

hs_2 = Holt(aapl_train, exponential=True).fit()
hs_forecast_2 = hs_2.forecast(test_length)

hs_3 = Holt(aapl_train, exponential=False, damped_trend=True).fit(damping_trend=0.99)
hs_forecast_3 = hs_3.forecast(test_length)

aapl.plot(color=COLORS[0], title="Holt's smoothing smoothing models", label = 'Actual', legend = True)

hs_forecast_1.plot(color=COLORS[1], legend = True, label='linear trend')
hs_1.fittedvalues.plot(color=[COLORS[1]])

hs_forecast_2.plot(color=COLORS[2], legend = True, label='exponential trend')
hs_2.fittedvalues.plot(color=[COLORS[2]])

hs_forecast_3.plot(color=COLORS[3], legend = True, label='exponential trend damped')
hs_3.fittedvalues.plot(color=[COLORS[3]])