# lag-modeling


# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
from imp import reload
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import os.path
import zipfile
import datetime
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.stats import linregress
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1)
red, blue, green = sns.color_palette("Set1", 3)

plt.warnings.simplefilter("ignore", category=FutureWarning)
plt.warnings.filterwarnings('ignore', message='axes.color_cycle is deprecated and replaced with axes.prop_cycle; please use the latter.')


# In[2]:

import lmfit
import curveball
import curveball.baranyi_roberts_model


# In[3]:

plate = pd.read_csv(r"C:\Users\Dana\Anaconda3\plate_templates\G-RG-R.csv")


# In[5]:

df = curveball.ioutils.read_tecan_xlsx('C:/Users/Dana/Downloads/Yoav_181115.xlsx', plate=plate, max_time=24)
df.head()


# In[6]:

curveball.plots.tsplot(df);


# In[7]:

m = curveball.models.fit_model(df[df.Strain=='G'], PLOT=False, PRINT=False)[0]
m


# In[8]:

lam1 = curveball.baranyi_roberts_model.lag(m)
lam1
# in the papaer lambda=1.714 for the same q0 and v


# In[9]:

lam2 = curveball.models.find_lag(m)
lam2


# In[10]:

ax = m.plot_fit(data_kws={'alpha':0}, fitfmt='-', init_kws={'ls':''})
ax.axvline(lam1, color='r')
ax.axvline(lam2, color='g')


# In[12]:

q0 = 0.05
v = 2.3
e = 2.71828
lags = {}
for i in range(0,4):
    q0_tag = q0*e**(v*i)
    lag1 = (curveball.baranyi_roberts_model.lag(model_result=None, q0=q0_tag,v=v))
    lags[q0_tag] = lag1
    
#lag2 = (curveball.models.find_lag(model_fit=None, q0=q0,v=v))
print(lags)
#print(lag2)


# In[13]:

#yvals = lags.values()
#xvals = lags.keys()
#plt.plot(yvals, xvals)

plt.plot([1.3237054077058361, 0.47840759069408556, 0.079645926446112583, 0.0086762447033672946], [0.05, 0.49870835118782497, 4.974200390889578, 49.613505508367524], 'ro')
plt.ylabel('lam')
plt.xlabel('q0')

plt.show()


# In[32]:

lags = {}
intervals = [0,0.5,1,1.5,2]
for i in intervals:
    q0_tag = q0*e**(v*i)
    lag1 = (curveball.baranyi_roberts_model.lag(model_result=None, q0=q0_tag,v=v))
    lags[q0_tag] = lag1
    
#lag2 = (curveball.models.find_lag(model_fit=None, q0=q0,v=v))
print(lags)
#print(lag2)


# In[24]:

# for different intervals

plt.plot([1.323, 0.866, 0.478, 0.214, 0.079], [0.05, 0.158, 0.498, 1.575, 4.974], 'ro')
plt.ylabel('lam')
plt.xlabel('q0')

plt.axis([0, 2, 0, 6])
plt.show()


# In[45]:

# for different intervals

#plt.plot([1.323, 0.866, 0.478, 0.214, 0.079], [0.05, 0.158, 0.498, 1.575, 4.974], 'ro')
plt.plot([1.323, 0.866, 0.478, 0.214], [0.05, 0.158, 0.498, 1.575], 'ro')
plt.ylabel('lam')
plt.xlabel('q0_tag')

plt.axis([0, 2, 0, 6])
plt.show()


# In[73]:

prior_time = np.linspace(0, 2)

slope, intercept, r_value, p_value, std_err = linregress([1.323, 0.866, 0.478, 0.214, 0.079], [0.05, 0.158, 0.498, 1.575, 4.974])
print("P-value is", p_value)
print("slope is", slope)
print("intercept is", intercept)
print("std_err is", std_err)
print("Coefficient of correlation is", r_value)
print("R^2 is", (r_value)**2)
      
lag_time = (intercept + slope * prior_time)

plt.plot(prior_time, lag_time) # prediction
plt.scatter([1.323, 0.866, 0.478, 0.214, 0.079], [0.05, 0.158, 0.498, 1.575, 4.974], marker='.') #data

plt.ylabel('lam')
plt.xlabel('q0_tag')


# In[75]:

# same but without smallest qo_tag
prior_time = np.linspace(0, 2)

slope, intercept, r_value, p_value, std_err = linregress([1.323, 0.866, 0.478, 0.214], [0.05, 0.158, 0.498, 1.575])
print("P-value is", p_value)
print("slope is", slope)
print("intercept is", intercept)
print("std_err is", std_err)
print("Coefficient of correlation is", r_value)
print("R^2 is", (r_value)**2)
      
lag_time = (intercept + slope * prior_time)

plt.plot(prior_time, lag_time) # prediction
plt.scatter([1.323, 0.866, 0.478, 0.214], [0.05, 0.158, 0.498, 1.575], marker='.') #data

plt.ylabel('lam')
plt.xlabel('q0_tag')

