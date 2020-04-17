#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


df=pd.read_csv('Salary_Data.csv')
print(df)


# In[4]:


df.head()


# In[23]:


x=df[['YearsExperience']].values
y=df[['Salary']].values
print(x)
print(y)


# In[25]:


from sklearn.linear_model import LinearRegression 
lr=LinearRegression()
lr.fit(x,y)


# In[36]:


ip=float(input("enter years of experience:"))
pred = lr.predict([[ip]])
print("your expected salary is "+" {:.2f}".format(float(pred)))


# 

# In[43]:


plt.scatter(x,y)
pred2=lr.predict(x)
plt.plot(x,pred2,color='red')
print(lr.coef_)


# In[46]:


from sklearn.metrics import r2_score
score=r2_score(y,pred2)
print(score)


# In[ ]:




