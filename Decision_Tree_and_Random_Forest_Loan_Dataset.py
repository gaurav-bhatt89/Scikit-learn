#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning - Supervised - Binary classification - Decision Tree + Random Forest

# In[1]:


## Lets import necessary python libraries
import numpy as np # Numerical Python library used to perform numerical operations
import pandas as pd # Pandas library used to handle and analyze structured data
import matplotlib.pyplot as plt # Plotting library
import seaborn as sns # Additional plotting library built over matplotlib (better visuals)
sns.set_theme(style='darkgrid',palette='viridis')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import string # This library will help us perform operations on string values


# In[3]:


## Lets import the loan data set and load it into a dataframe 
df = pd.read_csv(r"C:\Users\gmraj\Documents\Refactored_Py_DS_ML_Bootcamp-master\15-Decision-Trees-and-Random-Forests\loan_data.csv")


# In[4]:


df.info()


# In[9]:


## Exploratory Data Analysis
## Data Wrangling, Feature Consolidation/Engineering


# In[15]:


## Checking for missing values
plt.figure(figsize=(6,4))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title('Missing values will be highlighted by a yellow underline')
plt.show()


# In[17]:


## No missing value found


# In[23]:


## Lets check for correlation of the independent variables
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('Correlation Coefficient of Variables')
plt.show()


# In[29]:


sns.scatterplot(x='fico',y='int.rate',data=df)
plt.title('FICO Score vs Int. Rate [Correlation Coefficient Value: -0.71]')
plt.show()


# In[31]:


## As observed from the Heatmap, there is negative correlation with the coefficient value of -0.71 between FICO and Int. Rate
## As the FICO goes up, the Int. Rate goes down and vice verca


# In[102]:


df.head(3)


# In[140]:


## Lets create a new dataframe by tweaking existing one in such a way that we obtain
## purpose wise percentage of all the loans which are not fully paid


# In[122]:


purpose_groupby = df.groupby('purpose')['not.fully.paid'].describe()
purpose_groupby = pd.DataFrame(purpose_groupby)


# In[136]:


purpose_groupby = purpose_groupby.sort_values(by='mean',ascending=True)


# In[138]:


purpose_groupby


# In[143]:


## Here we can see that 'small_busines' category has highest amount of loans which are not fully paid
## where as 'major_purchase' and 'credit_card' has the highest amount of loans which are fully paid


# In[ ]:


## Lets now feature engineer the existing dataframe so it becomes ML algorithm ready


# In[145]:


## We will have to encode all the categorical variables so that all the data in the dataframe is in numerical values


# In[18]:


purpose = pd.get_dummies(df['purpose'],drop_first=True,dtype='float')
df.drop('purpose',axis=1,inplace=True)


# In[24]:


df = pd.concat([df,purpose],axis=1)


# In[19]:


df.head(5)


# In[28]:


## Standardize the dataset


# In[30]:


from sklearn.model_selection import train_test_split # to split the dataframe into a training and a testing set
from sklearn.preprocessing import StandardScaler # to standardize the dataframe values so that the values are standarized on a single x axis
from sklearn.tree import DecisionTreeClassifier # Decision Tree Algorithm (estimator)
from sklearn.pipeline import Pipeline # to automate the steps


# In[32]:


X = df.drop('not.fully.paid',axis=1)


# In[37]:


y = df['not.fully.paid']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[44]:


## Creating a pipeline that will first standardize the values and the pass them through the Decision Tree estimator
pipes = Pipeline([
    ('scaler',StandardScaler()),
    ('dec_tree',DecisionTreeClassifier())
])


# In[46]:


pipes.fit(X_train, y_train)


# In[152]:


## Importing metrics required for performance evaluation
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[50]:


predict_pipe = pipes.predict(X_test)


# In[52]:


print(classification_report(y_test,predict_pipe))
print('\n')
print(confusion_matrix(y_test,predict_pipe))


# ### Decision Tree estimator produced result of 74% accuracy

# ## Random Forest method

# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[65]:


pipes_rf = Pipeline([
    ('scaler_rf',StandardScaler()),
    ('rc_classifier',RandomForestClassifier(n_estimators=200))
])


# In[66]:


pipes_rf.fit(X_train, y_train)


# In[67]:


predict_rf = pipes_rf.predict(X_test)


# In[68]:


print(classification_report(y_test,predict_rf))
print('\n')
print(confusion_matrix(y_test,predict_rf))


# ### Random Forest estimator produced a result of 83% accuracy

# ## Thank You
