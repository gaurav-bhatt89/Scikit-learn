# This below Python code executed in Jupyter Notebook by [Anaconda Navigator](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.anaconda.com/products/navigator&ved=2ahUKEwiT5K_m_IuNAxWce_UHHVooNSwQFnoECBkQAQ&usg=AOvVaw2FiVm4Knmhe7xplbtYwLdO) 
## We will learn to 
  1. Load an existing publicly available ['Credit Card Defaults'](https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction) dataset
  2. Exploratory Data Analysis
  3. Data Cleansing / Wrangling
  4. Feature Consolidation / Engineering
  5. Applying Decision Tree estimator and evaluating its predction accuracy
  6. Applying Random Forest estimator and evaluating its prediction accuracy

### GitHub Notebook - [Link](https://github.com/gaurav-bhatt89/Scikit-learn/blob/main/Decision_Tree_and_Random_Forest_CCFraud_Dataset.ipynb)
### NBViewer - [Link](https://nbviewer.org/github/gaurav-bhatt89/Scikit-learn/blob/main/Decision_Tree_and_Random_Forest_CCFraud_Dataset.ipynb)
```python
## importing necessary python libraries
import numpy as np # numerical python library
import pandas as pd # create dataframes and perform operations
import string # perform string operations
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # beautify the charts and graphs
sns.set_theme(style='darkgrid',palette='Set2')

##this is so that we can have charts displayed between these block of codes on this notebook
%matplotlib inline

# lets import the dataset and load it into a dataframe
df = pd.read_csv(r"C:\Users\gmraj\Documents\Refactored_Py_DS_ML_Bootcamp-master\15-Decision-Trees-and-Random-Forests\Credit_Card_Fraud.csv")

## Exploratory Data Analysis
## Data Cleansing / Wrangling / Feature Engineering

## this dataset is moderate sized with > 500K indexes and over 20 feature columns
df.info()
```
![image](https://github.com/user-attachments/assets/d1e0a39a-1b6f-402a-9d1e-fd6f77ca128f)
```python
plt.figure(figsize=(12,5))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('Correlation Coefficients of Numerical Features')
plt.show()
```
![image](https://github.com/user-attachments/assets/f46769f6-5214-4847-8e1a-2a1f9cc61d35)
```python
## So far we have observed that only 'amt' has noticable positive correlation with 'is_fraud' of the coefficient 0.18
## Since many columns are Categorical, its possibe that they might give more truth later once we encode them

df.describe()
```
![image](https://github.com/user-attachments/assets/25e4b608-65b2-488c-ba04-b3eb58eaa4cc)
```python
plt.figure(figsize=(4,3))
sns.countplot(x='is_fraud',data=df)
plt.title('Count of Fraud vs Non Fraud entries')
plt.show()
```
![image](https://github.com/user-attachments/assets/020a12b2-7109-42ce-b771-194f8460c89a)
```python
## Lets check the overall count of fraud vs non fraud entries
df['is_fraud'].value_counts()
is_fraud
0    553574
1      2145
Name: count, dtype: int64

## Fraud entries account for 0.3860% of the overall entries
df['is_fraud'].describe()
count    555719.000000
mean          0.003860
std           0.062008
min           0.000000
25%           0.000000
50%           0.000000
75%           0.000000
max           1.000000
Name: is_fraud, dtype: float64

plt.figure(figsize=(4,3))
sns.countplot(x='is_fraud',data=df[df['is_fraud']==1],hue='gender')
plt.title('Gender wise CC Frauds')
plt.show()
```
![image](https://github.com/user-attachments/assets/5e2b6c32-9ded-4e46-965a-8e9a83674a0e)
```python
plt.figure(figsize=(4,3))
sns.countplot(x='is_fraud',data=df[df['is_fraud']==1],hue='category')
plt.legend(loc='best',bbox_to_anchor=[1,1,0.6,0.1])
plt.title('Category wise # of CC Frauds')
plt.show()
```
![image](https://github.com/user-attachments/assets/6ac96210-939f-4634-97d6-222b04968644)
```python
## Since we observed that 'enterntainment' and 'misc_pos' have significantly higher number of CC Frauds, lets check 
## the average ticket size of the transactions for each of these categories

plt.figure(figsize=(18,6))
sns.boxplot(x='category',y='amt',data=df)
plt.show()
```
![image](https://github.com/user-attachments/assets/f66811ba-9b85-4174-b91c-f392ad313e04)
```python
Since the above Box Plot is not comprehendable lets find out the mean of the amt column and then re-plot based on the adjusted amt values

df['amt'].describe()
count    555719.000000
mean         69.392810
std         156.745941
min           1.000000
25%           9.630000
50%          47.290000
75%          83.010000
max       22768.110000
Name: amt, dtype: float64

plt.figure(figsize=(18,6))
sns.boxplot(x='category',y='amt',data=df)
plt.ylim(0,150)
plt.title('IQR of Amount spent on each category')
plt.show()
```
![image](https://github.com/user-attachments/assets/47145bf7-6da9-4850-a9ea-e18ed6b52984)
```python
## Creating a new dataframe to store only those index where the is_fraud value is 1
df_fraud = df[df['is_fraud']==1]

df_fraud['amt'].describe()
count    2145.000000
mean      528.356494
std       392.747594
min         1.780000
25%       214.510000
50%       371.940000
75%       907.770000
max      1320.920000
Name: amt, dtype: float64

### We can oberve that for fraud transactions the mean ticket size of the transaction is significantly higher (69 vs 528)

plt.figure(figsize=(18,6))
sns.boxplot(x='category',y='amt',data=df_fraud)
# plt.ylim(0,150)
plt.title('IQR of Amount spent on each category for fraud transactions')
plt.show()
```
![image](https://github.com/user-attachments/assets/ae9ff9f5-2d27-4a94-b7c4-d1d8bd71c6b8)
```python
We observe that for enterntainment avg ticket size is ~50 usd and for misc. net its ~$15¶
Whereas for same categories for fraud transactions the avg ticket size is ~500 usd (10x) and ~800 usd (53x)¶

## Lets now check if Age is a factor in CC Frauds
## We will convert the existing dob column to pandas datatime 
df['dob'] = pd.to_datetime(df['dob'],dayfirst=True)

## Creating a new Age column based on difference between current year minus year of birth
df['age']=2025-df['dob'].dt.year

## Average age of the customers is 51.6 yrs
df['age'].describe()
count    555719.000000
mean         51.636237
std          17.418528
min          20.000000
25%          38.000000
50%          50.000000
75%          63.000000
max         101.000000
Name: age, dtype: float64

Average age of the customers who have been victim of the fraud is 53.72 yrs¶
df[df['is_fraud']==1]['age'].describe()
count    2145.000000
mean       53.738462
std        17.618287
min        23.000000
25%        39.000000
50%        53.000000
75%        66.000000
max       101.000000
Name: age, dtype: float64

## We are creating a new dataframe for only fraud victim customers. We will only include their age and fraud amount
df_age_fraud = df[df['is_fraud']==1][['age','amt']].sort_values(by='age',ascending=True)

plt.figure(figsize=(6,4))
sns.jointplot(x='age',y='amt',data=df_age_fraud,kind='hex')
plt.title('Age wise number of fraud cases')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/4c4d31c5-3de9-481e-9b32-103b48849ad9)
```python
Data is widely dispersed but we can clearly observe that maximum number of frauds are happening to customers aged between 35 and 50

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

df_age_fraud['age_bracket'] = df_age_fraud['age'].apply(my_age)

df_age_fraud.groupby('age_bracket')['amt'].describe()['mean']

age_bracket
20-30       608.721242
30-40       515.553701
40-50       457.945104
above 50    545.139908
Name: mean, dtype: float64

We can see that age 20-30 has the highest mean of fraud amount of $608 across all other categories.
It makes sense as this group is more likely to spend on enterntainment category which as 10x frauds amount wise

## Lets check if there is any correlation between frauds and the job of a customer
df_job_faud = df[df['is_fraud']==1][['gender','job','amt']]

df_job_faud['job'].nunique()
177

We have a total of 177 unique job titles which add upto a total of 2145 fraud cases

df_job_fraud_mean = df_job_faud.groupby('job').describe()['amt']['mean']

df_job_fraud_mean = pd.DataFrame(df_job_fraud_mean)

df_job_fraud_mean.sort_values(by='mean',ascending=False).head(10)
```
![image](https://github.com/user-attachments/assets/28c299ae-1dfe-4a0a-b2d2-926726f502d6)
```python
These are the top 5 job titles which have highest mean of fraudulent amount out of a total of 177 job titles

df_job_fraud_mean.sort_values(by='mean',ascending=True).head(10)
```
![image](https://github.com/user-attachments/assets/420fc94e-1cd5-41d3-9be5-e0106096f295)
```python
These are the top 5 job titles which have lowest mean of fraudulent amount out of a total of 177 job titles

## Lets check if there is any pattern that can be observed based on the day of the fraudelent transactons

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

df['trans_day'] = df['trans_date_trans_time'].dt.day_name()

df_day_fraud = df[df['is_fraud']==1][['trans_day']]

plt.figure(figsize=(6,4))
sns.countplot(x='trans_day',data=df_day_fraud)
plt.title('Number of Fraud transaction based on day of the week')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/85d02ce1-d10e-4dcb-bff3-47e5ea155aed)
```python
Now lets go ahead and Feature Engineer the data so that its ready to be fit into a model

## Encoding the categorical columns into numerical columns
category = pd.get_dummies(df['category'],drop_first=True,dtype='float')
job = pd.get_dummies(df['job'],drop_first=True,dtype='float')
trans_day = pd.get_dummies(df['trans_day'],drop_first=True,dtype='float')
gender = pd.get_dummies(df['gender'],drop_first=True,dtype='float')

df.drop(['trans_date_trans_time','cc_num','merchant','category','street','first','last','gender','street','city','state','job','dob','trans_num','unix_time','trans_day'],axis=1,inplace=True)

df = pd.concat([df,category,job,trans_day,gender],axis=1)

## Dropping first unnamed column
df = df.iloc[:, 1:]

X = df.drop('is_fraud',axis=1)

y = df['is_fraud']
```
![image](https://github.com/user-attachments/assets/56251c3b-8cc0-4fe8-9972-950f481916f4)
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

pipe_dec_tree = Pipeline([
    ('scaler',StandardScaler()),
    ('dec_tree',DecisionTreeClassifier(criterion='gini',splitter='best'))
])

pipe_dec_tree.fit(X_train,y_train)
```
![image](https://github.com/user-attachments/assets/019324aa-28b1-4bd1-8e9e-9569f2c6b460)
```python
predict_dec_tree = pipe_dec_tree.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(classification_report(y_test,predict_dec_tree))
print('\n')
print(confusion_matrix(y_test,predict_dec_tree))
print('\n')
print(accuracy_score(y_test,predict_dec_tree))
```
![image](https://github.com/user-attachments/assets/bf2cd25b-47f9-40b7-8ebe-2721193c6490)
```python
from sklearn.ensemble import RandomForestClassifier
pipe_rf = Pipeline([
    ('scaler',StandardScaler()),
    ('ran_for',RandomForestClassifier(criterion='gini',n_estimators=100))
])

pipe_rf.fit(X_train,y_train)
```
![image](https://github.com/user-attachments/assets/52455f06-fe52-4692-8ce1-0231c5261861)
```python
predict_rf = pipe_rf.predict(X_test)

print(classification_report(y_test,predict_rf))
print('\n')
print(confusion_matrix(y_test,predict_rf))
print('\n')
print(accuracy_score(y_test,predict_rf))
```
![image](https://github.com/user-attachments/assets/968f692f-f56f-45ee-805e-82cff4c9422b)
