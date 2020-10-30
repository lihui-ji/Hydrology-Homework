#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from numpy import nan
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import TimeSeriesSplit
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import warnings
warnings.simplefilter("ignore")
from skgstat import Variogram


# In[5]:


filename=["Ji_Lihui_OXFO_Monthly_Precip.txt","Ji_Lihui_DURH_Monthly_Precip.txt","Ji_Lihui_REED_Monthly_Precip.txt","Ji_Lihui_LAKE_Monthly_Precip.txt","Ji_Lihui_CLAY_Monthly_Precip.txt","Ji_Lihui_CLA2_Monthly_Precip.txt"]
varname=["OXFO","DURH","REED","CLA2","CLAY","LAKE"]
datalist=[]
temp=[]
for i in range(len(filename)):
    temp=pd.read_csv("C:\DUKE\courses\hydrology\HW3\Lihui_Ji_Files\\"+filename[i], sep='\t', skiprows=5)
    #temp['Date/Time (EST)'] = pd.to_datetime(temp['Date/Time (EST)'], format='%Y-%m-%d %H:%M:%S')
    temp[temp['Monthly SUM of Precipitation (in)'] < 0 ] = np.nan
    temp['Monthly SUM of Precipitation (in)']=temp['Monthly SUM of Precipitation (in)']*25.4
    temp.rename(columns = {'Monthly SUM of Precipitation (in)': 'Monthly Precipitation (mm)'}, inplace = True)
    datalist.append(temp['Monthly Precipitation (mm)'])


# In[6]:


x=datalist[0].cumsum()
y=datalist[1].cumsum()
z = np.polyfit(x, y, 1)
plt.plot(x,y,'o')
plt.plot(x,np.polyval(z, x),linewidth=2)
plt.title('double mass analysis between '+varname[0]+' and '+varname[1])
plt.grid()
plt.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\"+ 'double mass analysis between '+varname[0]+' and '+varname[1]+'.png', dpi = 300) 


# In[7]:


gamma=[]
for i in range(1,len(datalist)):
    gamma.append(0.5*np.mean((datalist[0]-datalist[i])**2))
h_abs=[37,56,64,71,80]
h_ns=[30,54,63,70,79]
h_ew=[21,11,5,12,14]
h_ns_avg=np.array(h_ns).mean()
h_ew_avg=np.array(h_ew).mean()


# In[8]:


average=[]
for i in range(0,len(datalist)):
    average.append(datalist[i].mean())
values=np.array(average)


# In[9]:


coords=np.array([[36.3,78.62],[36.03,78.68],[35.81,78.74],[35.73,78.68],[35.67,78.49],[35.59,78.46]])
V=Variogram(coords, values, normalize=False)
V.plot(hist=False)
print(V)
print('SemiVariogram Plot:')


# In[10]:


coords=np.array([[36.3,h_ew_avg],[36.03,h_ew_avg],[35.81,h_ew_avg],[35.73,h_ew_avg],[35.67,h_ew_avg],[35.59,h_ew_avg]])
V=Variogram(coords, values, normalize=False)
V.plot(hist=False)
print('SemiVariogram in N-S Direction')
print(V)
print('SemiVariogram Plot:')


# In[11]:


coords=np.array([[h_ns_avg,78.62],[h_ns_avg,78.68],[h_ns_avg,78.74],[h_ns_avg,78.68],[h_ns_avg,78.49],[h_ns_avg,78.46]])
V=Variogram(coords, values, normalize=False)
V.plot(hist=False)
print('SemiVariogram in E-W Direction')
print(V)
print('SemiVariogram Plot:')


# In[12]:


values


# In[16]:


OK = OrdinaryKriging(
    np.array([36.3,36.03,35.81,35.73,35.67,35.59]),
    np.array([78.62,78.86,78.74,78.68,78.49,78.46]),
    values,
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
)
gridx = np.arange(35.5, 36.5, 0.1)
gridy = np.arange(78, 79, 0.1)
z, ss = OK.execute("grid", gridx, gridy)
fig1=plt.figure(figsize=(10, 6))
plt.imshow(z,interpolation='nearest',cmap='viridis')
plt.title('Oridinary Kringing')
plt.colorbar()
fig1.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+' Kringing.png', dpi = 300)
fig2=plt.figure(figsize=(10, 6))
plt.imshow(ss,interpolation='nearest',cmap='viridis')
plt.title('Variance ')
plt.colorbar()
fig2.savefig("C:\DUKE\courses\hydrology\HW3\output figures\\ "+' Kringing_Variance.png', dpi = 300)


# In[ ]:




