#!/usr/bin/env python
# coding: utf-8

# In[3]:


#predictions using Supervised ML
#Yash Bhis

import pandas as pd
import numpy as np
import matplotlib.pyplot as mt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

url = "http://bit.ly/w-data"
file = pd.read_csv(url)
print("Done Importing")


# In[4]:


file.head(5)


# In[10]:


file.describe()


# In[8]:


file.shape


# In[11]:


correlation=file.corr()
correlation


# In[12]:


file.nunique()


# In[13]:


file.plot(x='Hours', y='Scores' , style='o')
mt.title('Hours Vs Scores')
mt.xlabel('Hours')
mt.ylabel('Scores')
mt.show()


# In[14]:


x=file.iloc[:, :-1].values
y=file.iloc[:, 1].values


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)


# In[17]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)
print("Completed Training ----- Time for Testing")


# In[18]:


line = regressor.coef_*x+regressor.intercept_  
mt.scatter(x, y)  
mt.plot(x, line);  
mt.show()


# In[19]:


print(regressor.intercept_)


# In[20]:


print(regressor.coef_)


# In[21]:


y_pred = regressor.predict(x_test)


# In[22]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df


# In[23]:


hours = [[9.25]]
own_pred = regressor.predict(hours)  
print("Number of hours =",hours )  
print("Prediction Score = ",own_pred)


# In[ ]:




