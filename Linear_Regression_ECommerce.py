#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Machine Learning - Supervised - Linear Regression


# In[3]:


## Importing Libraries


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import string


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


## Lets import the data and load it into a dataframe


# In[11]:


df = pd.read_csv(r"C:\Users\gmraj\Documents\Refactored_Py_DS_ML_Bootcamp-master\11-Linear-Regression\Ecommerce Customers.csv")


# In[13]:


## EDA + Data Wrangling + Feature Engineering


# In[15]:


df.info()


# In[21]:


df.describe().transpose()


# In[29]:


## Lets check the correlations of the numeric values


# In[27]:


plt.figure(figsize=(4,3))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()


# In[31]:


## Length of Membership, Time on App and Avg. Session Length have (in decreasing order) positive correlations with Yearly Spent


# In[35]:


plt.figure(figsize=(4,3))
sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=df)
plt.show()


# In[37]:


plt.figure(figsize=(4,3))
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)
plt.show()


# In[54]:


## Lets check for missing values


# In[58]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[60]:


## No Missing Values


# In[62]:


## Lets proceed with Feature Engineering. We will wrangle the data from Categorical columns and try to obtain some meaningful insights


# In[181]:


email = df['Email'].apply(lambda x: x.split('@')[1])
email = pd.DataFrame(email)
email = email['Email'].apply(lambda x: x.split('.')[0])
email = pd.DataFrame(email)


# In[19]:


df.head(5)


# In[173]:


## Lets create dummy variables for Email and Avatar so that we can use it into ML algorithm 


# In[195]:


# pd.get_dummies(email['Email'],drop_first=True,dtype='float')
# pd.get_dummies(df['Avatar'],drop_first=True,dtype='float')


# In[197]:


## After creating dummy variable the overall unique values in Email and Avatar are approx 50% of total rows. Hence we wont be using them


# In[201]:


## Lets drop unnecessary columns


# In[203]:


df.drop(['Email','Address','Avatar'],axis=1,inplace=True)


# In[205]:


df.head(5)


# In[207]:


from sklearn.model_selection import train_test_split


# In[209]:


df.columns


# In[211]:


X = df[['Avg. Session Length', 'Time on App', 'Time on Website',
       'Length of Membership']]
y = df['Yearly Amount Spent']


# In[213]:


from sklearn.linear_model import LinearRegression


# In[262]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
lr = LinearRegression()
lr.fit(X_train,y_train)
predictions_i = lr.predict(X_test)   


# In[272]:


plt.scatter(y_test,predictions_i)
plt.show()


# In[274]:


sns.displot(y_test-predictions_i,kde=True)
plt.show()


# In[276]:


from sklearn import metrics


# In[278]:


print(metrics.root_mean_squared_error(y_test,predictions_i))


# In[280]:


lr.intercept_


# In[286]:


pd.DataFrame(lr.coef_,X.columns,columns=['Coef'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




