import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import seaborn as sns


def convert_quarter_to_timestamp(date):
    if re.match(r'(\d+)(Q\d)', date):
        return dates.date2num(pd.to_datetime(re.sub(r'(\d+)(Q\d)', r'\1-\2', date)))
    return date


@plt.FuncFormatter
def fake_dates(x, pos):
    """ Custom formatter to turn floats into e.g., 2016-05-08"""
    return dates.num2date(x).strftime('%Y-%m-%d')


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

fig, ax = plt.subplots()
sns.regplot('Quarter', 'GDP', data=melt_data, ax=ax)

ax.xaxis.set_major_formatter(fake_dates)
ax.tick_params(labelrotation=45)
fig.tight_layout()

plt.show()
