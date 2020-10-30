#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from numpy import nan
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import TimeSeriesSplit


# In[53]:


filename=["Ji_Lihui_WAYN_Hourly_Precip.txt","Ji_Lihui_LAKE_Hourly_Precip.txt","Ji_Lihui_PLYM_Hourly_Precip.txt"]
varname=["WYAN","LAKE","PLYM"]
datalist=[]
for i in range(len(filename)):
    temp=pd.read_csv("C:\DUKE\courses\hydrology\HW3\Lihui_Ji_Files\\"+filename[i], sep='\t', skiprows=5)
    temp['Date/Time (EST)'] = pd.to_datetime(temp['Date/Time (EST)'], format='%Y-%m-%d %H:%M:%S')
    temp[temp['Hourly Precipitation (in)'] < 0 ] = np.nan
    temp['Hourly Precipitation (in)']=temp['Hourly Precipitation (in)']*25.4
    temp[temp['Hourly Precipitation From Impact Sensor (in)'] < 0 ] = np.nan
    temp['Hourly Precipitation From Impact Sensor (in)']=temp['Hourly Precipitation From Impact Sensor (in)']*25.4
    temp.rename(columns = {'Hourly Precipitation (in)': 'Hourly Precipitation (mm)',                            'Hourly Precipitation From Impact Sensor (in)': 'Hourly Precipitation From Impact Sensor (mm)'}, inplace = True)
    datalist.append(temp)


# In[54]:


i=2


# In[55]:


datalist[i].head()


# In[56]:


precipitation_hour=datalist[i]
precipitation_day=precipitation_hour.groupby([precipitation_hour['Date/Time (EST)'].dt.year, precipitation_hour['Date/Time (EST)'].dt.month,                                        precipitation_hour['Date/Time (EST)'].dt.day]).sum()
precipitation_day.index=precipitation_day.index.set_names(['year', 'month', 'day'])
precipitation_day.reset_index(inplace=True)
precipitation_day.set_index(pd.to_datetime(precipitation_day[['year', 'month', 'day']]),inplace=True)
precipitation_day.rename(columns = {'Hourly Precipitation (mm)': 'Daily Precipitation (mm)',                            'Hourly Precipitation From Impact Sensor (mm)': 'Daily Precipitation From Impact Sensor (mm)'}, inplace = True)


# In[57]:


precipitation_day_mean=precipitation_day.groupby([precipitation_day['month'],precipitation_day['day']]).mean().drop(columns=['year'])
precipitation_day_mean.reset_index(inplace=True)
precipitation_day_mean=precipitation_day_mean.set_index(pd.DatetimeIndex(pd.date_range(start='1/1/2000', end='12/31/2000').values))
#just assign 2000 to include all possible days, it is averaged year value not certain year


# In[58]:


precipitation_day_mean.head()


# In[59]:


precipitation_month_max=precipitation_day_mean.groupby([precipitation_day_mean['month']]).max().drop(columns=['day'])
precipitation_month_max.reset_index(inplace=True)
precipitation_month_max.rename(columns = {'Daily Precipitation (mm)': 'Max Monthly Precipitation (mm/d)',                            'Daily Precipitation From Impact Sensor (mm)': 'Max Monthly Precipitation From Impact Sensor (mm/d)'}, inplace = True)


# In[60]:


precipitation_month_max.head()


# In[61]:


varname[i]


# In[62]:


fig, ax = plt.subplots(figsize=(16, 8));
ax.plot(precipitation_day_mean[['Daily Precipitation (mm)', 'Daily Precipitation From Impact Sensor (mm)']]);
date_form = DateFormatter("%b-%d");
ax.xaxis.set_major_locator(mdates.MonthLocator());
ax.xaxis.set_major_formatter(date_form);
ax.set_title('Diurnal cycle for Each Month on a Monthly Basis of '+ varname[i],fontsize=20, pad=20)
plt.xticks(rotation=60);
plt.xlabel('time',fontsize=15)
plt.ylabel('precipation(mm)',fontsize=15)
plt.grid(True)
plt.legend( ['Traditional Sensor', 'Impact Sensor'], prop={"size":15})
fig.tight_layout(pad=5.0)
save_results_to = '/Users/S/Desktop/Results/'
plt.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+varname[i] + ' Diurnal cycle for Each Month on a Monthly Basis.png', dpi = 300)


# In[63]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16, 6));
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax1.bar(range(12), precipitation_month_max['Max Monthly Precipitation (mm/d)'], tick_label=labels)
ax1.set_axisbelow(True)
ax1.grid(axis='y',linestyle='--', linewidth=1)
ax1.set_title('Traditional Sensor',fontsize=15)
ax1.set_xlabel('Month',fontsize=15)
ax1.set_ylabel('Precipation(mm/d)',fontsize=15)

ax2.bar(range(12), precipitation_month_max['Max Monthly Precipitation From Impact Sensor (mm/d)'], tick_label=labels)
ax2.set_axisbelow(True)
ax2.grid(axis='y',linestyle='--', linewidth=1)
ax2.set_title('Impact Sensor',fontsize=15)
ax2.set_xlabel('Month',fontsize=15)
ax2.set_ylabel('Precipation(mm/d)',fontsize=15)
fig.suptitle('Maximum Rainfall Rate at '+ varname[i],fontsize=20)
fig.tight_layout(pad=5.0)
plt.show()
fig.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+varname[i] + ' Maximum Rainfall Rate.png', dpi = 300)


# In[64]:


precipitation_hour.head()


# In[65]:


first_non_NaN_index=precipitation_hour['Hourly Precipitation From Impact Sensor (mm)'].first_valid_index()
first_non_NaN_date=precipitation_hour['Date/Time (EST)'][first_non_NaN_index]
start_time=first_non_NaN_date.replace(hour=0) + pd.offsets.MonthBegin(1)
precipitation_hour_noNaN=precipitation_hour[precipitation_hour['Date/Time (EST)']>=start_time]


# In[66]:


precipitation_hour.head()


# In[67]:


precipitation_month=precipitation_hour_noNaN.groupby([datalist[0]['Date/Time (EST)'].dt.year, datalist[0]['Date/Time (EST)'].dt.month]).sum()
precipitation_month.index=precipitation_month.index.set_names(['year', 'month'])
precipitation_month['day']=15
precipitation_month.reset_index(inplace=True)
precipitation_month.set_index(pd.to_datetime(precipitation_month[['year', 'month','day']]),inplace=True)
precipitation_month.rename(columns = {'Hourly Precipitation (mm)': 'Monthly Precipitation (mm)',                            'Hourly Precipitation From Impact Sensor (mm)': 'Monthly Precipitation From Impact Sensor (mm)'}, inplace = True)
precipitation_month.drop(columns=['day'],inplace=True)


# In[68]:


mean_monthly_trad=precipitation_month['Monthly Precipitation (mm)'].mean()
mean_monthly_impact=precipitation_month['Monthly Precipitation From Impact Sensor (mm)'].mean()
precipitation_month['Anomalies From Traditional Sensor(mm)']=precipitation_month['Monthly Precipitation (mm)']-mean_monthly_trad
precipitation_month['Anomalies From Impact Sensor(mm)']=precipitation_month['Monthly Precipitation From Impact Sensor (mm)']-mean_monthly_impact
precipitation_month['Cumulative Anomalies From Traditional Sensor(mm)']=precipitation_month['Anomalies From Traditional Sensor(mm)'].cumsum()
precipitation_month['Cumulative Anomalies From Impact Sensor(mm)']=precipitation_month['Anomalies From Impact Sensor(mm)'].cumsum()

precipitation_month['Normalized Anomalies From Traditional Sensor(mm)']=precipitation_month['Anomalies From Traditional Sensor(mm)']/max(abs(precipitation_month['Anomalies From Traditional Sensor(mm)']))
precipitation_month['Normalized Anomalies From Impact Sensor(mm)']=precipitation_month['Anomalies From Impact Sensor(mm)']/max(abs(precipitation_month['Anomalies From Impact Sensor(mm)']))
precipitation_month['Normalized Cumulative Anomalies From Traditional Sensor(mm)']=precipitation_month['Cumulative Anomalies From Traditional Sensor(mm)']/max(abs(precipitation_month['Cumulative Anomalies From Traditional Sensor(mm)']))
precipitation_month['Normalized Cumulative Anomalies From Impact Sensor(mm)']=precipitation_month['Cumulative Anomalies From Impact Sensor(mm)']/max(abs(precipitation_month['Cumulative Anomalies From Impact Sensor(mm)']))


# In[69]:


precipitation_month.describe()


# In[70]:


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(16, 10));


ax1.plot(precipitation_month[['Normalized Anomalies From Traditional Sensor(mm)', 'Normalized Cumulative Anomalies From Traditional Sensor(mm)']]);
ax1.set_axisbelow(True)
ax1.grid(axis='both',linestyle='-', linewidth=1)
ax1.set_title('Traditional Sensor',fontsize=15)
ax1.xaxis.set_major_locator(mdates.YearLocator());
ax1.set_ylabel('Normalized Anomalies',fontsize=15)
ax1.axhline(0, color='gray', linewidth=2)
ax1.set_xlabel('Year',fontsize=15)

ax2.plot(precipitation_month[['Normalized Anomalies From Impact Sensor(mm)', 'Normalized Cumulative Anomalies From Impact Sensor(mm)']]);
ax2.set_axisbelow(True)
ax2.grid(axis='both',linestyle='-', linewidth=1)
ax2.set_title('Impact Sensor',fontsize=15)
ax2.xaxis.set_major_locator(mdates.YearLocator());
ax2.set_ylabel('Normalized Anomalies',fontsize=15)
ax2.axhline(0, color='gray', linewidth=2)
ax2.set_xlabel('Year',fontsize=15)
fig.suptitle('Monthly RainFall Anomalies at '+ varname[i],fontsize=20)
fig.tight_layout(pad=5.0)
plt.show()
fig.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+varname[i] + ' Monthly RainFall Anomalies.png', dpi = 300)


# In[71]:


year_min=precipitation_hour['Date/Time (EST)'].dt.year.min()
year_median=precipitation_hour['Date/Time (EST)'].dt.year.median()
year_max=precipitation_hour['Date/Time (EST)'].dt.year.max()
precipitation_hour_whole_period=precipitation_hour
precipitation_hour_first_half_period=precipitation_hour.loc[precipitation_hour['Date/Time (EST)'].dt.year.between(year_min,year_median)]
precipitation_hour_second_half_period=precipitation_hour.loc[precipitation_hour['Date/Time (EST)'].dt.year.between(year_median+1,year_max)]


# In[72]:


precipitation_hour_annually=precipitation_hour_whole_period.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_annually=precipitation_hour_annually.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_annually.reset_index(inplace=True)
precipitation_hour_annually.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_annually['Year'] = precipitation_hour_annually['Year'].astype(int)
precipitation_hour_annually['Rank']=precipitation_hour_annually['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_annually['Weibull Return Perood']=(precipitation_hour_annually['Rank'].max()+1)/precipitation_hour_annually['Rank']
precipitation_hour_annually.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_annually['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_annually['Hourly Precipitation (mm)'].values
model_annually = LinearRegression()
model_annually.fit(x, y)
para_lambda_annually=0.15*model_annually.coef_
para_psi_annually=model_annually.intercept_/para_lambda_annually+1/0.15

precipitation_hour_Jan=precipitation_hour_whole_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(1,1)]
precipitation_hour_Jan=precipitation_hour_Jan.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_Jan=precipitation_hour_Jan.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_Jan.reset_index(inplace=True)
precipitation_hour_Jan.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_Jan['Year'] = precipitation_hour_Jan['Year'].astype(int)
precipitation_hour_Jan['Rank']=precipitation_hour_Jan['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_Jan['Weibull Return Perood']=(precipitation_hour_Jan['Rank'].max()+1)/precipitation_hour_Jan['Rank']
precipitation_hour_Jan.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_Jan['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_Jan['Hourly Precipitation (mm)'].values
model_Jan = LinearRegression()
model_Jan.fit(x, y)
para_lambda_Jan=0.15*model_Jan.coef_
para_psi_Jan=model_Jan.intercept_/para_lambda_Jan+1/0.15

precipitation_hour_May=precipitation_hour_whole_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(5,5)]
precipitation_hour_May=precipitation_hour_May.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_May=precipitation_hour_May.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_May.reset_index(inplace=True)
precipitation_hour_May.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_May['Year'] = precipitation_hour_May['Year'].astype(int)
precipitation_hour_May['Rank']=precipitation_hour_May['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_May['Weibull Return Perood']=(precipitation_hour_May['Rank'].max()+1)/precipitation_hour_May['Rank']
precipitation_hour_May.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_May['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_May['Hourly Precipitation (mm)'].values
model_May = LinearRegression()
model_May.fit(x, y)
para_lambda_May=0.15*model_May.coef_
para_psi_May=model_May.intercept_/para_lambda_May+1/0.15


precipitation_hour_Sep=precipitation_hour_whole_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(1,1)]
precipitation_hour_Sep=precipitation_hour_Sep.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_Sep=precipitation_hour_Sep.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_Sep.reset_index(inplace=True)
precipitation_hour_Sep.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_Sep['Year'] = precipitation_hour_Sep['Year'].astype(int)
precipitation_hour_Sep['Rank']=precipitation_hour_Sep['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_Sep['Weibull Return Perood']=(precipitation_hour_Sep['Rank'].max()+1)/precipitation_hour_Sep['Rank']
precipitation_hour_Sep.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_Sep['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_Sep['Hourly Precipitation (mm)'].values
model_Sep = LinearRegression()
model_Sep.fit(x, y)
para_lambda_Sep=0.15*model_Sep.coef_
para_psi_Sep=model_Sep.intercept_/para_lambda_Sep+1/0.15

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,figsize=(16, 24),constrained_layout=False);
fig.tight_layout(pad=5.0)
fig.suptitle('Whole Period Recurrence Interval of '+ varname[i],fontsize=30,y=1)
x=np.array([10,20,50,100])

ax1.plot(precipitation_hour_annually['Weibull Return Perood'],precipitation_hour_annually['Hourly Precipitation (mm)'],'o-');
ax1.plot(np.linspace(0.0, 110.0),model_annually.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_annually=model_annually.predict(np.power(x,0.15).reshape((-1,1)))
ax1.plot(x,y_annually,'s');
for a,b in zip(x, y_annually): 
    ax1.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax1.set_xscale("log")
ax1.set_ylim((0, y_annually.max()+10))
ax1.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax1.tick_params(axis='both', labelsize=15)
ax1.xaxis.set_major_formatter(ScalarFormatter());
ax1.set_title('Annually Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax1.set_xlabel('Return Period/year',fontsize=15)
ax1.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax1.xaxis.labelpad = 15
ax1.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax1.grid(True)
fig.tight_layout(pad=5.0)

ax2.plot(precipitation_hour_Jan['Weibull Return Perood'],precipitation_hour_Jan['Hourly Precipitation (mm)'],'o-');
ax2.plot(np.linspace(0.0, 110.0),model_Jan.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_Jan=model_Jan.predict(np.power(x,0.15).reshape((-1,1)))
ax2.plot(x,y_Jan,'s');
for a,b in zip(x, y_Jan): 
    ax2.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax2.set_xscale("log")
ax2.set_ylim((0, y_Jan.max()+10))
ax2.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax2.tick_params(axis='both', labelsize=15)
ax2.xaxis.set_major_formatter(ScalarFormatter());
ax2.set_title('January Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax2.set_xlabel('Return Period/year',fontsize=15)
ax2.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax2.xaxis.labelpad = 15
ax2.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax2.grid(True)

ax3.plot(precipitation_hour_May['Weibull Return Perood'],precipitation_hour_May['Hourly Precipitation (mm)'],'o-');
ax3.plot(np.linspace(0.0, 110.0),model_May.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_May=model_May.predict(np.power(x,0.15).reshape((-1,1)))
ax3.plot(x,y_May,'s');
for a,b in zip(x, y_May): 
    ax3.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax3.set_xscale("log")
ax3.set_ylim((0, y_May.max()+10))
ax3.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax3.tick_params(axis='both', labelsize=15)
ax3.xaxis.set_major_formatter(ScalarFormatter());
ax3.set_title('May Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax3.set_xlabel('Return Period/year',fontsize=15)
ax3.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax3.xaxis.labelpad = 15
ax3.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax3.grid(True)

ax4.plot(precipitation_hour_Sep['Weibull Return Perood'],precipitation_hour_Sep['Hourly Precipitation (mm)'],'o-');
ax4.plot(np.linspace(0.0, 110.0),model_Sep.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_Sep=model_Sep.predict(np.power(x,0.15).reshape((-1,1)))
ax4.plot(x,y_Sep,'s');
for a,b in zip(x, y_Sep): 
    ax4.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax4.set_xscale("log")
ax4.set_ylim((0, y_Sep.max()+10))
ax4.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax4.tick_params(axis='both', labelsize=15)
ax4.xaxis.set_major_formatter(ScalarFormatter());
ax4.set_title('September Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax4.set_xlabel('Return Period/year',fontsize=15)
ax4.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax4.xaxis.labelpad = 15
ax4.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax4.grid(True)

fig.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+varname[i] + ' Whole Period Recurrence Interval.png', dpi = 300)
plt.show()


# In[73]:


precipitation_hour_annually=precipitation_hour_first_half_period.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_annually=precipitation_hour_annually.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_annually.reset_index(inplace=True)
precipitation_hour_annually.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_annually['Year'] = precipitation_hour_annually['Year'].astype(int)
precipitation_hour_annually['Rank']=precipitation_hour_annually['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_annually['Weibull Return Perood']=(precipitation_hour_annually['Rank'].max()+1)/precipitation_hour_annually['Rank']
precipitation_hour_annually.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_annually['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_annually['Hourly Precipitation (mm)'].values
model_annually = LinearRegression()
model_annually.fit(x, y)
para_lambda_annually=0.15*model_annually.coef_
para_psi_annually=model_annually.intercept_/para_lambda_annually+1/0.15

precipitation_hour_Jan=precipitation_hour_first_half_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(1,1)]
precipitation_hour_Jan=precipitation_hour_Jan.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_Jan=precipitation_hour_Jan.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_Jan.reset_index(inplace=True)
precipitation_hour_Jan.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_Jan['Year'] = precipitation_hour_Jan['Year'].astype(int)
precipitation_hour_Jan['Rank']=precipitation_hour_Jan['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_Jan['Weibull Return Perood']=(precipitation_hour_Jan['Rank'].max()+1)/precipitation_hour_Jan['Rank']
precipitation_hour_Jan.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_Jan['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_Jan['Hourly Precipitation (mm)'].values
model_Jan = LinearRegression()
model_Jan.fit(x, y)
para_lambda_Jan=0.15*model_Jan.coef_
para_psi_Jan=model_Jan.intercept_/para_lambda_Jan+1/0.15

precipitation_hour_May=precipitation_hour_first_half_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(5,5)]
precipitation_hour_May=precipitation_hour_May.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_May=precipitation_hour_May.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_May.reset_index(inplace=True)
precipitation_hour_May.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_May['Year'] = precipitation_hour_May['Year'].astype(int)
precipitation_hour_May['Rank']=precipitation_hour_May['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_May['Weibull Return Perood']=(precipitation_hour_May['Rank'].max()+1)/precipitation_hour_May['Rank']
precipitation_hour_May.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_May['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_May['Hourly Precipitation (mm)'].values
model_May = LinearRegression()
model_May.fit(x, y)
para_lambda_May=0.15*model_May.coef_
para_psi_May=model_May.intercept_/para_lambda_May+1/0.15


precipitation_hour_Sep=precipitation_hour_first_half_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(1,1)]
precipitation_hour_Sep=precipitation_hour_Sep.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_Sep=precipitation_hour_Sep.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_Sep.reset_index(inplace=True)
precipitation_hour_Sep.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_Sep['Year'] = precipitation_hour_Sep['Year'].astype(int)
precipitation_hour_Sep['Rank']=precipitation_hour_Sep['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_Sep['Weibull Return Perood']=(precipitation_hour_Sep['Rank'].max()+1)/precipitation_hour_Sep['Rank']
precipitation_hour_Sep.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_Sep['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_Sep['Hourly Precipitation (mm)'].values
model_Sep = LinearRegression()
model_Sep.fit(x, y)
para_lambda_Sep=0.15*model_Sep.coef_
para_psi_Sep=model_Sep.intercept_/para_lambda_Sep+1/0.15

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,figsize=(16, 24),constrained_layout=False);
fig.tight_layout(pad=5.0)
fig.suptitle('First Half Period Recurrence Interval of '+ varname[i],fontsize=30,y=1)
x=np.array([10,20,50,100])

ax1.plot(precipitation_hour_annually['Weibull Return Perood'],precipitation_hour_annually['Hourly Precipitation (mm)'],'o-');
ax1.plot(np.linspace(0.0, 110.0),model_annually.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_annually=model_annually.predict(np.power(x,0.15).reshape((-1,1)))
ax1.plot(x,y_annually,'s');
for a,b in zip(x, y_annually): 
    ax1.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax1.set_xscale("log")
ax1.set_ylim((0, 100))
ax1.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax1.tick_params(axis='both', labelsize=15)
ax1.xaxis.set_major_formatter(ScalarFormatter());
ax1.set_title('Annually Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax1.set_xlabel('Return Period/year',fontsize=15)
ax1.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax1.xaxis.labelpad = 15
ax1.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax1.grid(True)
fig.tight_layout(pad=5.0)

ax2.plot(precipitation_hour_Jan['Weibull Return Perood'],precipitation_hour_Jan['Hourly Precipitation (mm)'],'o-');
ax2.plot(np.linspace(0.0, 110.0),model_Jan.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_Jan=model_Jan.predict(np.power(x,0.15).reshape((-1,1)))
ax2.plot(x,y_Jan,'s');
for a,b in zip(x, y_Jan): 
    ax2.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax2.set_xscale("log")
ax2.set_ylim((0, y_Jan.max()+10))
ax2.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax2.tick_params(axis='both', labelsize=15)
ax2.xaxis.set_major_formatter(ScalarFormatter());
ax2.set_title('January Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax2.set_xlabel('Return Period/year',fontsize=15)
ax2.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax2.xaxis.labelpad = 15
ax2.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax2.grid(True)

ax3.plot(precipitation_hour_May['Weibull Return Perood'],precipitation_hour_May['Hourly Precipitation (mm)'],'o-');
ax3.plot(np.linspace(0.0, 110.0),model_May.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_May=model_May.predict(np.power(x,0.15).reshape((-1,1)))
ax3.plot(x,y_May,'s');
for a,b in zip(x, y_May): 
    ax3.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax3.set_xscale("log")
ax3.set_ylim((0, y_May.max()+10))
ax3.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax3.tick_params(axis='both', labelsize=15)
ax3.xaxis.set_major_formatter(ScalarFormatter());
ax3.set_title('May Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax3.set_xlabel('Return Period/year',fontsize=15)
ax3.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax3.xaxis.labelpad = 15
ax3.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax3.grid(True)

ax4.plot(precipitation_hour_Sep['Weibull Return Perood'],precipitation_hour_Sep['Hourly Precipitation (mm)'],'o-');
ax4.plot(np.linspace(0.0, 110.0),model_Sep.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_Sep=model_Sep.predict(np.power(x,0.15).reshape((-1,1)))
ax4.plot(x,y_Sep,'s');
for a,b in zip(x, y_Sep): 
    ax4.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax4.set_xscale("log")
ax4.set_ylim((0, y_Sep.max()+10))
ax4.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax4.tick_params(axis='both', labelsize=15)
ax4.xaxis.set_major_formatter(ScalarFormatter());
ax4.set_title('September Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax4.set_xlabel('Return Period/year',fontsize=15)
ax4.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax4.xaxis.labelpad = 15
ax4.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax4.grid(True)

fig.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+varname[i] + ' First Half Period Recurrence Interval.png', dpi = 300)
plt.show()


# In[74]:


precipitation_hour_annually=precipitation_hour_second_half_period.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_annually=precipitation_hour_annually.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_annually.reset_index(inplace=True)
precipitation_hour_annually.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_annually['Year'] = precipitation_hour_annually['Year'].astype(int)
precipitation_hour_annually['Rank']=precipitation_hour_annually['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_annually['Weibull Return Perood']=(precipitation_hour_annually['Rank'].max()+1)/precipitation_hour_annually['Rank']
precipitation_hour_annually.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_annually['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_annually['Hourly Precipitation (mm)'].values
model_annually = LinearRegression()
model_annually.fit(x, y)
para_lambda_annually=0.15*model_annually.coef_
para_psi_annually=model_annually.intercept_/para_lambda_annually+1/0.15

precipitation_hour_Jan=precipitation_hour_second_half_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(1,1)]
precipitation_hour_Jan=precipitation_hour_Jan.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_Jan=precipitation_hour_Jan.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_Jan.reset_index(inplace=True)
precipitation_hour_Jan.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_Jan['Year'] = precipitation_hour_Jan['Year'].astype(int)
precipitation_hour_Jan['Rank']=precipitation_hour_Jan['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_Jan['Weibull Return Perood']=(precipitation_hour_Jan['Rank'].max()+1)/precipitation_hour_Jan['Rank']
precipitation_hour_Jan.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_Jan['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_Jan['Hourly Precipitation (mm)'].values
model_Jan = LinearRegression()
model_Jan.fit(x, y)
para_lambda_Jan=0.15*model_Jan.coef_
para_psi_Jan=model_Jan.intercept_/para_lambda_Jan+1/0.15

precipitation_hour_May=precipitation_hour_second_half_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(5,5)]
precipitation_hour_May=precipitation_hour_May.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_May=precipitation_hour_May.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_May.reset_index(inplace=True)
precipitation_hour_May.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_May['Year'] = precipitation_hour_May['Year'].astype(int)
precipitation_hour_May['Rank']=precipitation_hour_May['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_May['Weibull Return Perood']=(precipitation_hour_May['Rank'].max()+1)/precipitation_hour_May['Rank']
precipitation_hour_May.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_May['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_May['Hourly Precipitation (mm)'].values
model_May = LinearRegression()
model_May.fit(x, y)
para_lambda_May=0.15*model_May.coef_
para_psi_May=model_May.intercept_/para_lambda_May+1/0.15


precipitation_hour_Sep=precipitation_hour_second_half_period.loc[precipitation_hour['Date/Time (EST)'].dt.month.between(1,1)]
precipitation_hour_Sep=precipitation_hour_Sep.groupby([precipitation_hour['Date/Time (EST)'].dt.year]).max()['Hourly Precipitation (mm)']
precipitation_hour_Sep=precipitation_hour_Sep.replace(0,nan).dropna(how='all',axis=0).to_frame()
precipitation_hour_Sep.reset_index(inplace=True)
precipitation_hour_Sep.rename(columns={'Date/Time (EST)':'Year'},inplace=True)
precipitation_hour_Sep['Year'] = precipitation_hour_Sep['Year'].astype(int)
precipitation_hour_Sep['Rank']=precipitation_hour_Sep['Hourly Precipitation (mm)'].rank(ascending=False).astype(int)
precipitation_hour_Sep['Weibull Return Perood']=(precipitation_hour_Sep['Rank'].max()+1)/precipitation_hour_Sep['Rank']
precipitation_hour_Sep.sort_values(by='Weibull Return Perood', ascending=True,inplace=True)
x = np.power(precipitation_hour_Sep['Weibull Return Perood'].values,0.15).reshape((-1, 1))
y = precipitation_hour_Sep['Hourly Precipitation (mm)'].values
model_Sep = LinearRegression()
model_Sep.fit(x, y)
para_lambda_Sep=0.15*model_Sep.coef_
para_psi_Sep=model_Sep.intercept_/para_lambda_Sep+1/0.15

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,figsize=(16, 24),constrained_layout=False);
fig.tight_layout(pad=5.0)
fig.suptitle('Second Half Period Recurrence Interval of '+ varname[i],fontsize=30,y=1)
x=np.array([10,20,50,100])

ax1.plot(precipitation_hour_annually['Weibull Return Perood'],precipitation_hour_annually['Hourly Precipitation (mm)'],'o-');
ax1.plot(np.linspace(0.0, 110.0),model_annually.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_annually=model_annually.predict(np.power(x,0.15).reshape((-1,1)))
ax1.plot(x,y_annually,'s');
for a,b in zip(x, y_annually): 
    ax1.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax1.set_xscale("log")
ax1.set_ylim((0, 100))
ax1.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax1.tick_params(axis='both', labelsize=15)
ax1.xaxis.set_major_formatter(ScalarFormatter());
ax1.set_title('Annually Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax1.set_xlabel('Return Period/year',fontsize=15)
ax1.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax1.xaxis.labelpad = 15
ax1.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax1.grid(True)
fig.tight_layout(pad=5.0)

ax2.plot(precipitation_hour_Jan['Weibull Return Perood'],precipitation_hour_Jan['Hourly Precipitation (mm)'],'o-');
ax2.plot(np.linspace(0.0, 110.0),model_Jan.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_Jan=model_Jan.predict(np.power(x,0.15).reshape((-1,1)))
ax2.plot(x,y_Jan,'s');
for a,b in zip(x, y_Jan): 
    ax2.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax2.set_xscale("log")
ax2.set_ylim((0, y_Jan.max()+10))
ax2.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax2.tick_params(axis='both', labelsize=15)
ax2.xaxis.set_major_formatter(ScalarFormatter());
ax2.set_title('January Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax2.set_xlabel('Return Period/year',fontsize=15)
ax2.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax2.xaxis.labelpad = 15
ax2.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax2.grid(True)

ax3.plot(precipitation_hour_May['Weibull Return Perood'],precipitation_hour_May['Hourly Precipitation (mm)'],'o-');
ax3.plot(np.linspace(0.0, 110.0),model_May.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_May=model_May.predict(np.power(x,0.15).reshape((-1,1)))
ax3.plot(x,y_May,'s');
for a,b in zip(x, y_May): 
    ax3.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax3.set_xscale("log")
ax3.set_ylim((0, y_May.max()+10))
ax3.set_xticks([1, 2, 5, 10, 20, 50 , 100])
ax3.tick_params(axis='both', labelsize=15)
ax3.xaxis.set_major_formatter(ScalarFormatter());
ax3.set_title('May Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax3.set_xlabel('Return Period/year',fontsize=15)
ax3.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax3.xaxis.labelpad = 15
ax3.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax3.grid(True)

ax4.plot(precipitation_hour_Sep['Weibull Return Perood'],precipitation_hour_Sep['Hourly Precipitation (mm)'],'o-');
ax4.plot(np.linspace(0.0, 110.0),model_Sep.predict(np.power(np.linspace(0.0, 110.0),0.15).reshape((-1,1))));
y_Sep=model_Sep.predict(np.power(x,0.15).reshape((-1,1)))
ax4.plot(x,y_Sep,'s');
for a,b in zip(x, y_Sep): 
    ax4.text(a-1, b*1.05, str(int(b)), fontsize=15)
ax4.set_xscale("log")
ax4.set_ylim((0, y_Sep.max()+10))
ax4.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax4.tick_params(axis='both', labelsize=15)
ax4.xaxis.set_major_formatter(ScalarFormatter());
ax4.set_title('September Average Recurrence Interval of '+ varname[i],fontsize=20, pad=20)
ax4.set_xlabel('Return Period/year',fontsize=15)
ax4.set_ylabel('Hourly precipation(mm)',fontsize=15)
ax4.xaxis.labelpad = 15
ax4.legend( ['Emperical Frequency', 'GEV - II'], prop={"size":15},loc='upper left')
ax4.grid(True)

fig.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+varname[i] + ' Second Half Period Recurrence Interval.png', dpi = 300)
plt.show()


# In[75]:


temp=precipitation_hour
temp=temp.groupby([precipitation_hour['Date/Time (EST)'].dt.year, precipitation_hour['Date/Time (EST)'].dt.month]).sum()
temp.index=temp.index.set_names(['year', 'month'])
temp.reset_index(inplace=True)
precipitation_hour_Hurst_sep=[]
for year_index in np.array(range(2000,2020)):
    precipitation_hour_Hurst_sep.append(temp[temp['year']==year_index]['Hourly Precipitation (mm)'].values)


# In[76]:


Ch=[]
h=[3,5,10]
logh=np.log(h)
mean=precipitation_hour_Hurst_sep[0].mean()
std=np.std(precipitation_hour_Hurst_sep[0])
N=len(precipitation_hour_Hurst_sep[0])
for j in [2,4,9]:
    Ch.append(np.inner(precipitation_hour_Hurst_sep[0]-mean, precipitation_hour_Hurst_sep[j+1]-mean)/std/std/(N-1))
logCh=np.log(Ch)


# In[ ]:


model_Hurst = LinearRegression()
model_Hurst.fit(logh.reshape((-1, 1)), logCh)
beta=abs(model_Hurst.coef_)
print(beta)


# In[ ]:





# In[ ]:




