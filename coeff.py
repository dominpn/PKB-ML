import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('corr.csv', sep=':', decimal=",")
data.describe(include='all')
# data.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median()
# for col in data.columns.values:
#     if data[col].isnull().sum() == 0:
#         continue
#     if col == 'Climate':
#         guess_values = data.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
#     else:
#         guess_values = data.groupby('Region')[col].median()
#     for region in data['Region'].unique():
#         data[col].loc[(data[col].isnull())&(data['Region']==region)] = guess_values[region]

plt.figure(figsize=(16,12))
sns.heatmap(data=data.iloc[:,2:].corr(method='spearman'),annot=True,fmt='.2f',cmap='coolwarm')
# plt.show()
plt.savefig('spearman2.png')

# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20,20))
# plt.subplots_adjust(hspace=0.4)
#
# corr_to_gdp = pd.Series()
# for col in data.columns.values[2:]:
#     if ((col!='GDP ($ per capita)')&(col!='Climate')):
#         corr_to_gdp[col] = data['GDP ($ per capita)'].corr(data[col], method='spearman')
# abs_corr_to_gdp = corr_to_gdp.abs().sort_values(ascending=False)
# corr_to_gdp = corr_to_gdp.loc[abs_corr_to_gdp.index]
#
# for i in range(len(corr_to_gdp)):
#     plt.figure(figsize=(16, 12))
#     sns.regplot(x=corr_to_gdp.index.values[i], y='GDP ($ per capita)', data=data, fit_reg=False, marker='x')
#     title = 'correlation=' + str(corr_to_gdp[i])
#     plt.title(title)
#     plt.savefig(f'{title}.png')
#
# for i in range(3):
#     for j in range(3):
#         sns.regplot(x=corr_to_gdp.index.values[i*3+j], y='GDP ($ per capita)', data=data,
#                    ax=axes[i,j], fit_reg=True, marker='.')
#         title = 'correlation='+str(corr_to_gdp[i*3+j])
#         axes[i,j].set_title(title)
# axes[1,2].set_xlim(0,102)
# # plt.show()
# plt.savefig('temp2.png')
