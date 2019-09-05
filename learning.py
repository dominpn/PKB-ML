import re

import pandas as pd

import math
import numpy as np
import quandl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style, dates
import datetime
from dateutil.relativedelta import relativedelta

style.use('ggplot')


def convert_quarter_to_timestamp(date):
    if re.match(r'(\d+)(Q\d)', date):
        return pd.to_datetime(re.sub(r'(\d+)(Q\d)', r'\1-\2', date))
    return date


@plt.FuncFormatter
def fake_dates(x, pos):
    """ Custom formatter to turn floats into e.g., 2016-05-08"""
    if type(x) is float:
        return dates.num2date(x).strftime('%Y-%m-%d')


style.use('ggplot')

gdp_data = pd.read_csv('gdp2.csv', encoding="ISO-8859-13", thousands=',')
gdp_data = gdp_data.replace(':', '')
gdp_data = gdp_data.replace(',', '.')
gdp_data = gdp_data.rename(columns=lambda x: convert_quarter_to_timestamp(x))
columns = gdp_data.columns[1:].values
gdp_data[columns] = gdp_data[columns].apply(pd.to_numeric)
gdp_data = gdp_data.dropna(axis='columns', how='all')

pv2 = pd.pivot_table(gdp_data, index=['Country'], dropna=False, fill_value=0.0)
pv2.columns = gdp_data.columns[1:].values

plot_data = pv2.T.reset_index()
plot_data.rename(columns={'index': 'Quarter'}, inplace=True)
# unpivot the data, change from table view, where we have columns for each
# country, to big long time series data, [year, country code, value]
melt_data = pd.melt(plot_data, id_vars=['Quarter'], var_name='Country')
melt_data.rename(columns={'value': 'GDP'}, inplace=True)

forecast_col = 'GDP'
melt_data.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(melt_data)))
melt_data['GDP'] = melt_data[forecast_col].shift(-forecast_out)

X = np.array(melt_data.drop(['GDP', 'Country'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

melt_data.dropna(inplace=True)

y = np.array(melt_data['GDP'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
melt_data['Forecast'] = np.nan

last_date = melt_data.iloc[-1]['Quarter']
next_date = last_date + relativedelta(months=3)

for i in forecast_set:
    print(f'Predticted value {i}')
    next_date = next_date + relativedelta(months=3)
    melt_data.loc[next_date] = [np.nan for _ in range(len(melt_data.columns)-1)]+[i]

# na wykresie pojawi się cokolwiek w momencie gdy przewidziane zostanie przewidziane wiecej predykcji
melt_data['GDP'].plot()
melt_data['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
