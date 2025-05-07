#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning - Supervised - Binary Classification - Decision Tree and Random Forest using Pipelines

# In[4]:


## importing necessary python libraries
import numpy as np # numerical python library
import pandas as pd # create dataframes and perform operations
import string # perform string operations
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # beautify the charts and graphs
sns.set_theme(style='darkgrid',palette='Set2')


# In[14]:


##this is so that we can have charts displayed between these block of codes on this notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


# lets import the dataset and load it into a dataframe
df = pd.read_csv(r"C:\Users\gmraj\Documents\Refactored_Py_DS_ML_Bootcamp-master\15-Decision-Trees-and-Random-Forests\Credit_Card_Fraud.csv")


# In[20]:


## Exploratory Data Analysis
## Data Cleansing / Wrangling / Feature Engineering


# In[26]:


## this dataset is moderate sized with > 500K indexes and over 20 feature columns
df.info()


# In[34]:


plt.figure(figsize=(12,5))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('Correlation Coefficients of Numerical Features')
plt.show()


# In[36]:


## So far we have observed that only 'amt' has noticable positive correlation with 'is_fraud' of the coefficient 0.18
## Since many columns are Categorical, its possibe that they might give more truth later once we encode them


# In[38]:


df.describe()


# In[46]:


plt.figure(figsize=(4,3))
sns.countplot(x='is_fraud',data=df)
plt.title('Count of Fraud vs Non Fraud entries')
plt.show()


# In[58]:


## Lets check the overall count of fraud vs non fraud entries
df['is_fraud'].value_counts()


# In[62]:


## Fraud entries account for 0.3860% of the overall entries
df['is_fraud'].describe()


# In[76]:


plt.figure(figsize=(4,3))
sns.countplot(x='is_fraud',data=df[df['is_fraud']==1],hue='gender')
plt.title('Gender wise CC Frauds')
plt.show()


# In[92]:


plt.figure(figsize=(4,3))
sns.countplot(x='is_fraud',data=df[df['is_fraud']==1],hue='category')
plt.legend(loc='best',bbox_to_anchor=[1,1,0.6,0.1])
plt.title('Category wise # of CC Frauds')
plt.show()


# In[94]:


## Since we observed that 'enterntainment' and 'misc_pos' have significantly higher number of CC Frauds, lets check 
## the average ticket size of the transactions for each of these categories


# In[105]:


plt.figure(figsize=(18,6))
sns.boxplot(x='category',y='amt',data=df)
plt.show()


# ### Since the above Box Plot is not comprehendable lets find out the mean of the amt column and then re-plot based on the adjusted amt values

# In[111]:


df['amt'].describe()


# In[109]:


plt.figure(figsize=(18,6))
sns.boxplot(x='category',y='amt',data=df)
plt.ylim(0,150)
plt.title('IQR of Amount spent on each category')
plt.show()


# In[117]:


## Creating a new dataframe to store only those index where the is_fraud value is 1
df_fraud = df[df['is_fraud']==1]


# In[121]:


df_fraud['amt'].describe()


# ### We can oberve that for fraud transactions the mean ticket size of the transaction is significantly higher (69 vs 528)

# In[125]:


plt.figure(figsize=(18,6))
sns.boxplot(x='category',y='amt',data=df_fraud)
# plt.ylim(0,150)
plt.title('IQR of Amount spent on each category for fraud transactions')
plt.show()


# ### We observe that for enterntainment avg ticket size is ~50 usd and for misc. net its ~$15
# ### Whereas for same categories for fraud transactions the avg ticket size is ~500 usd (10x) and ~800 usd (53x)

# In[133]:


## Lets now check if Age is a factor in CC Frauds
## We will convert the existing dob column to pandas datatime 
df['dob'] = pd.to_datetime(df['dob'],dayfirst=True)


# In[141]:


## Creating a new Age column based on difference between current year minus year of birth
df['age']=2025-df['dob'].dt.year


# In[145]:


## Average age of the customers is 51.6 yrs
df['age'].describe()


# ### Average age of the customers who have been victim of the fraud is 53.72 yrs

# In[151]:


df[df['is_fraud']==1]['age'].describe()


# In[193]:


## We are creating a new dataframe for only fraud victim customers. We will only include their age and fraud amount
df_age_fraud = df[df['is_fraud']==1][['age','amt']].sort_values(by='age',ascending=True)


# In[205]:


plt.figure(figsize=(6,4))
sns.jointplot(x='age',y='amt',data=df_age_fraud,kind='hex')
plt.title('Age wise number of fraud cases')
plt.tight_layout()
plt.show()


# ### Data is widely dispersed but we can clearly observe that maximum number of frauds are happening to customers aged between 35 and 50

# In[250]:


## Lets create a cohort of age so that we can better understand which age group is more susceptible to CC fraud attempts
def my_age(x):
    if (x>=20) & (x<30):
        return '20-30'
    elif (x>=30) & (x<40):
        return '30-40'
    elif (x>=40) & (x<50):
        return '40-50'
    elif x<20:
        return 'under 20'
    else:
        return 'above 50'


# In[252]:


df_age_fraud['age_bracket'] = df_age_fraud['age'].apply(my_age)


# In[268]:


df_age_fraud.groupby('age_bracket')['amt'].describe()['mean']


# ### We can see that age 20-30 has the highest mean of fraud amount of $608 across all other categories.
# ### It makes sense as this group is more likely to spend on enterntainment category which as 10x frauds amount wise

# In[294]:


## Lets check if there is any correlation between frauds and the job of a customer
df_job_faud = df[df['is_fraud']==1][['gender','job','amt']]


# In[305]:


df_job_faud['job'].nunique()


# ### We have a total of 177 unique job titles which add upto a total of 2145 fraud cases

# In[323]:


df_job_fraud_mean = df_job_faud.groupby('job').describe()['amt']['mean']


# In[327]:


df_job_fraud_mean = pd.DataFrame(df_job_fraud_mean)


# In[337]:


df_job_fraud_mean.sort_values(by='mean',ascending=False).head(10)


# ### These are the top 5 job titles which have highest mean of fraudulent amount out of a total of 177 job titles

# In[341]:


df_job_fraud_mean.sort_values(by='mean',ascending=True).head(10)


# ### These are the top 5 job titles which have lowest mean of fraudulent amount out of a total of 177 job titles

# In[352]:


## Lets check if there is any pattern that can be observed based on the day of the fraudelent transactons


# In[360]:


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])


# In[366]:


df['trans_day'] = df['trans_date_trans_time'].dt.day_name()


# In[370]:


df.head(3)


# In[374]:


df_day_fraud = df[df['is_fraud']==1][['trans_day']]


# In[388]:


plt.figure(figsize=(6,4))
sns.countplot(x='trans_day',data=df_day_fraud)
plt.title('Number of Fraud transaction based on day of the week')
plt.tight_layout()
plt.show()


# ### Now lets go ahead and Feature Engineer the data so that its ready to be fit into a model

# In[401]:


## Encoding the categorical columns into numerical columns
category = pd.get_dummies(df['category'],drop_first=True,dtype='float')
job = pd.get_dummies(df['job'],drop_first=True,dtype='float')
trans_day = pd.get_dummies(df['trans_day'],drop_first=True,dtype='float')
gender = pd.get_dummies(df['gender'],drop_first=True,dtype='float')


# In[423]:


df.drop(['trans_date_trans_time','cc_num','merchant','category','street','first','last','gender','street','city','state','job','dob','trans_num','unix_time','trans_day'],axis=1,inplace=True)


# In[427]:


df = pd.concat([df,category,job,trans_day,gender],axis=1)


# In[442]:


## Dropping first unnamed column
df = df.iloc[:, 1:] 


# In[444]:


df.head(3)


# In[448]:


X = df.drop('is_fraud',axis=1)


# In[456]:


y = df['is_fraud']


# ## Decision Tree implementation

# In[459]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# In[461]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[463]:


pipe_dec_tree = Pipeline([
    ('scaler',StandardScaler()),
    ('dec_tree',DecisionTreeClassifier(criterion='gini',splitter='best'))
])


# In[465]:


pipe_dec_tree.fit(X_train,y_train)


# In[467]:


predict_dec_tree = pipe_dec_tree.predict(X_test)


# In[469]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[471]:


print(classification_report(y_test,predict_dec_tree))
print('\n')
print(confusion_matrix(y_test,predict_dec_tree))
print('\n')
print(accuracy_score(y_test,predict_dec_tree))


# ### Decison Tree Model predicted the outcomes with an accuracy of 99.73%

# ## Now lets run the same model on Random Forest estimator 

# In[478]:


from sklearn.ensemble import RandomForestClassifier


# In[480]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[482]:


pipe_rf = Pipeline([
    ('scaler',StandardScaler()),
    ('ran_for',RandomForestClassifier(criterion='gini',n_estimators=100))
])


# In[484]:


pipe_rf.fit(X_train,y_train)


# In[487]:


predict_rf = pipe_rf.predict(X_test)


# In[488]:


print(classification_report(y_test,predict_rf))
print('\n')
print(confusion_matrix(y_test,predict_rf))
print('\n')
print(accuracy_score(y_test,predict_rf))


# ### Decison Tree Model predicted the outcomes with an accuracy of 99.79%

# In[ ]:


## Thank You


# In[ ]:





# In[ ]:




