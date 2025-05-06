#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning - Supervised - Binary Classification - Log Reg

# In[4]:


## importing libraries


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="muted")


# In[10]:


import string
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


## Lets import the data and load it into the dataframe


# In[14]:


## Lets import the loan data set and load it into a dataframe 
df = pd.read_csv(r"C:\Users\gmraj\Documents\Refactored_Py_DS_ML_Bootcamp-master\13-Logistic-Regression\loan_data.csv")


# In[16]:


## Exploratory Data Analysis
## Data Cleansing, Wrangling
## Feture Cleanup, Engineering


# In[18]:


df.info()


# In[24]:


df.describe().transpose()


# In[28]:


## Lets check correlation 
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()


# In[32]:


## Lets check the overall age distribution of the applicants


# In[42]:


plt.figure(figsize=(4,3))
sns.displot(x='person_age',data=df,kde=True)
plt.title('Overall Age distribution of the applicants')
plt.show()


# In[44]:


## Lets check overall Loan Approved Vs Rejected Rate


# In[48]:


plt.figure(figsize=(4,3))
sns.countplot(x='loan_status',data=df)
plt.title('Loan Rejected vs Approved')
plt.show()


# In[50]:


## Lets check the Loan Rejected vs Approved by different parameters 


# In[54]:


plt.figure(figsize=(4,3))
sns.countplot(x='loan_status',data=df,hue='person_gender')
plt.title('Gender wise loan status')
plt.show()


# In[90]:


plt.figure(figsize=(4,3))
sns.countplot(x='loan_status',data=df,hue='person_home_ownership')
plt.title('Home Ownership wise loan status')
plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()


# In[92]:


plt.figure(figsize=(4,3))
sns.countplot(x='person_home_ownership',data=df)
plt.title('Home Ownership wise # of applications')
# plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()


# In[136]:


by_ownership = df.groupby('person_home_ownership')['loan_status'].describe()
by_ownership = pd.DataFrame(by_ownership)
by_ownership.sort_values(by='mean',ascending=True)
by_ownership


# In[112]:


plt.figure(figsize=(6,3))
sns.countplot(x='loan_status',data=df,hue='loan_intent')
plt.title('Loan Intent wise loan status')
plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()


# In[134]:


ln_intent = df.groupby('loan_intent')['loan_status'].describe()
ln_intent = pd.DataFrame(ln_intent)
ln_intent.sort_values(by='mean',ascending=True)


# In[148]:


plt.figure(figsize=(6,3))
sns.boxplot(x='loan_status',y='credit_score',data=df)
plt.title('Credit Score by Loan Status')
# plt.ylim(575,700)
# plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()


# In[153]:


plt.figure(figsize=(6,3))
sns.boxplot(x='person_education',y='loan_amnt',data=df)
plt.title('Credit Score by Loan Status')
# plt.ylim(575,700)
# plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()


# In[155]:


plt.figure(figsize=(6,3))
sns.countplot(x='loan_status',data=df,hue='previous_loan_defaults_on_file')
plt.title('Loan Status by PL Defaults')
plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()


# In[169]:


gender = pd.get_dummies(df['person_gender'],drop_first=True,dtype='float')
education = pd.get_dummies(df['person_education'],drop_first=True,dtype='float')
home_ownership = pd.get_dummies(df['person_home_ownership'],drop_first=True,dtype='float')
loan_intent = pd.get_dummies(df['loan_intent'],drop_first=True,dtype='float')
defaults = pd.get_dummies(df['previous_loan_defaults_on_file'],drop_first=True,dtype='float')


# In[ ]:


df.drop(['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file'],axis=1,inplace=True)


# In[184]:


df = pd.concat([df,gender,education,home_ownership,loan_intent,defaults],axis=1)


# In[186]:


df.head(5)


# In[270]:


# plt.figure(figsize=(15,6))
# sns.clustermap(df.corr(numeric_only=True),annot=True)
# plt.show()


# In[188]:


df.columns


# In[190]:


X = df[['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score', 'male', 'Bachelor', 'Doctorate',
       'High School', 'Master', 'OTHER', 'OWN', 'RENT', 'EDUCATION',
       'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE','Yes']]

y = df['loan_status']


# In[192]:


from sklearn.model_selection import train_test_split


# In[286]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# In[287]:


from sklearn.linear_model import LogisticRegression


# In[288]:


lr = LogisticRegression(solver='saga',max_iter=10000000,verbose=3)


# In[289]:


lr.fit(X_train,y_train)


# In[290]:


predict = lr.predict(X_test)


# In[291]:


from sklearn.metrics import classification_report,confusion_matrix


# In[292]:


print(classification_report(y_test,predict))
print('\n')
print(confusion_matrix(y_test,predict))


# ## Pipeline

# In[303]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[377]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)


# In[378]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('classifier', LogisticRegression(random_state=42))  # Logistic Regression model
])


# In[379]:


pipeline.fit(X_train, y_train)


# In[380]:


y_pred = pipeline.predict(X_test)


# In[381]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[382]:


print(classification_report(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test,y_pred))


# ## Thank You

# In[ ]:




