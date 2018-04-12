# Python, import your variables
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Python, read your data
ubc = pd.read_csv('UBC.csv')
ubc['Adj_Close_Log'] = np.log(ubc['Adj Close'])
ubc.head()

# Plotting stuff
# plt.plot(ubc['Adj Close'])
# plt.plot(ubc['Adj_Close_Log'])

# Define your model and fit to data
df_ubc = ubc[['Date','Adj_Close_Log']]
df_ubc.columns = ['ds','y']
m = Prophet()
m.fit(df_ubc);

# Python, create future series place holder
future = m.make_future_dataframe(periods=365)
future.tail()

# Python, predit future
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Python
m.plot(forecast);
