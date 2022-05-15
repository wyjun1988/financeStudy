#prophetTest.py
from matplotlib.pyplot import title
import pandas as pd
import seaborn as sns
from fbprophet import Prophet

import yfinance as yf
from matplotlib import pyplot as plt


df = yf.download(tickers='AAPL', start='2011-01-01', end='2020-12-31', progress=False)
df.reset_index(drop=False, inplace=True)
df.rename(columns={'Date':'ds', 'Close':'y'}, inplace=True)
print(df)

train_indices = df.ds.apply(lambda x: x.year) < 2019
df_train = df.loc[train_indices].dropna()
df_test = df.loc[~train_indices].reset_index(drop=True)

model_prophet = Prophet(seasonality_mode='additive')
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_prophet.fit(df_test)

df_future = model_prophet.make_future_dataframe(periods=365)
df_pred=model_prophet.predict(df_future)
model_prophet.plot(df_pred)

selected_columns = ['ds', 'yhat_lower', 'yhat_upper', 'yhat']

df_pred = df_pred.loc[:, selected_columns].reset_index(drop=True)
df_test = df_test.merge(df_pred, on=['ds'], how='left')
df_test.ds = pd.to_datetime(df_test.ds)
df_test.set_index('ds',inplace=True)

fig, ax = plt.subplots(1,1)

ax = sns.lineplot(data=df_test[['y', 'yhat_lower', 'yhat_upper', 'yhat']])
ax.fill_between(df_test.index, df_test.yhat_lower, df_test.yhat_upper, alpha=0.3)
ax.set(title= 'AAPL actual vs predicted', xlabel='Date', ylabel='Price')