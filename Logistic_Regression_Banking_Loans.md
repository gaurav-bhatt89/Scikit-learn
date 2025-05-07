# This below Python code executed in Jupyter Notebook by [Anaconda Navigator](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.anaconda.com/products/navigator&ved=2ahUKEwiT5K_m_IuNAxWce_UHHVooNSwQFnoECBkQAQ&usg=AOvVaw2FiVm4Knmhe7xplbtYwLdO) 
## We will learn to 
  1. Load an existing publicly available ['Loan Status'](https://github.com/gaurav-bhatt89/Datasets/blob/main/Loan_Data.csv) dataset
  2. Exploratory Data Analysis
  3. Data Cleansing / Wrangling
  4. Feature Consolidation / Engineering
  5. Applying Logistic Regression estimator and evaluating its predction accuracy
     
### GitHub Notebook - [Link](https://github.com/gaurav-bhatt89/Scikit-learn/blob/main/Decision_Tree_and_Random_Forest_Loan_Dataset.ipynb)
### NBViewer - Pending
```python
## importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="muted")

import string
%matplotlib inline

## Lets import the data and load it into the dataframe

## Lets import the loan data set and load it into a dataframe 
df = pd.read_csv(r"C:\Users\gmraj\Documents\Refactored_Py_DS_ML_Bootcamp-master\13-Logistic-Regression\loan_data.csv")

## Exploratory Data Analysis
## Data Cleansing, Wrangling
## Feture Cleanup, Engineering

df.info()
```
![image](https://github.com/user-attachments/assets/24f19332-b196-4bad-85ff-7cbf1a3baeda)
```python
df.describe().transpose()
```
![image](https://github.com/user-attachments/assets/801ee8c7-3924-4ab1-8037-ed66cbe2494a)
```python
## Lets check correlation 
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()
```
![image](https://github.com/user-attachments/assets/3b323fe1-4c49-4d66-ac39-fbee774443f9)
```python
## Lets check the overall age distribution of the applicants
plt.figure(figsize=(4,3))
sns.displot(x='person_age',data=df,kde=True)
plt.title('Overall Age distribution of the applicants')
plt.show()
```
![image](https://github.com/user-attachments/assets/912551a8-6b1b-4a75-98f1-cb2948119cb9)
```python
## Lets check overall Loan Approved Vs Rejected Rate
plt.figure(figsize=(4,3))
sns.countplot(x='loan_status',data=df)
plt.title('Loan Rejected vs Approved')
plt.show()
```
![image](https://github.com/user-attachments/assets/d79cee3e-451a-41e9-8678-e12cda6ddfef)
```python
## Lets check the Loan Rejected vs Approved by different parameters
plt.figure(figsize=(4,3))
sns.countplot(x='loan_status',data=df,hue='person_gender')
plt.title('Gender wise loan status')
plt.show()
```
![image](https://github.com/user-attachments/assets/010e2a04-777b-455f-8c8b-2a42e5b1670c)
```python
plt.figure(figsize=(4,3))
sns.countplot(x='loan_status',data=df,hue='person_home_ownership')
plt.title('Home Ownership wise loan status')
plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()
```
![image](https://github.com/user-attachments/assets/c5e54719-1563-4d01-bf5d-99377cd8ba49)
```python
plt.figure(figsize=(4,3))
sns.countplot(x='person_home_ownership',data=df)
plt.title('Home Ownership wise # of applications')
# plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()
```
![image](https://github.com/user-attachments/assets/51526dca-bd5e-497d-9cf3-53230e321ea3)
```python
by_ownership = df.groupby('person_home_ownership')['loan_status'].describe()
by_ownership = pd.DataFrame(by_ownership)
by_ownership.sort_values(by='mean',ascending=True)
by_ownership
```
![image](https://github.com/user-attachments/assets/2e7a5336-2958-465e-80d1-0e47498fb3cf)
```python
plt.figure(figsize=(6,3))
sns.countplot(x='loan_status',data=df,hue='loan_intent')
plt.title('Loan Intent wise loan status')
plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()
```
![image](https://github.com/user-attachments/assets/5c13471d-9c5a-4780-9d86-e05226693077)
```python
ln_intent = df.groupby('loan_intent')['loan_status'].describe()
ln_intent = pd.DataFrame(ln_intent)
ln_intent.sort_values(by='mean',ascending=True)
```
![image](https://github.com/user-attachments/assets/1a645344-5583-4c00-9c69-bd0a8a793bd6)
```python
plt.figure(figsize=(6,3))
sns.boxplot(x='loan_status',y='credit_score',data=df)
plt.title('Credit Score by Loan Status')
# plt.ylim(575,700)
# plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()
```
![image](https://github.com/user-attachments/assets/974848d9-6244-4595-8ffc-731f6e4ab5d7)
```python
plt.figure(figsize=(6,3))
sns.boxplot(x='person_education',y='loan_amnt',data=df)
plt.title('Credit Score by Loan Status')
# plt.ylim(575,700)
# plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()
```
![image](https://github.com/user-attachments/assets/b80b5013-d77a-479c-83ac-bb4ae2c356c1)
```python
plt.figure(figsize=(6,3))
sns.countplot(x='loan_status',data=df,hue='previous_loan_defaults_on_file')
plt.title('Loan Status by PL Defaults')
plt.legend(loc='best', bbox_to_anchor=(1,1,0.5,0.1))
plt.show()
```
![image](https://github.com/user-attachments/assets/421017f1-c27a-4db3-b8fa-b49d1b3446b6)
```python
gender = pd.get_dummies(df['person_gender'],drop_first=True,dtype='float')
education = pd.get_dummies(df['person_education'],drop_first=True,dtype='float')
home_ownership = pd.get_dummies(df['person_home_ownership'],drop_first=True,dtype='float')
loan_intent = pd.get_dummies(df['loan_intent'],drop_first=True,dtype='float')
defaults = pd.get_dummies(df['previous_loan_defaults_on_file'],drop_first=True,dtype='float')

df.drop(['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file'],axis=1,inplace=True)

df = pd.concat([df,gender,education,home_ownership,loan_intent,defaults],axis=1)

df.head(5)
```
![image](https://github.com/user-attachments/assets/a91e6b80-bffb-4102-8278-5f7433d79c7b)
```python
# plt.figure(figsize=(15,6))
# sns.clustermap(df.corr(numeric_only=True),annot=True)
# plt.show()
df.columns
Index(['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score', 'loan_status', 'male', 'Bachelor', 'Doctorate',
       'High School', 'Master', 'OTHER', 'OWN', 'RENT', 'EDUCATION',
       'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE', 'Yes'],
      dtype='object')

X = df[['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score', 'male', 'Bachelor', 'Doctorate',
       'High School', 'Master', 'OTHER', 'OWN', 'RENT', 'EDUCATION',
       'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE','Yes']]

y = df['loan_status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='saga',max_iter=10000000,verbose=3)
lr.fit(X_train,y_train)
```
![image](https://github.com/user-attachments/assets/1f274aae-0e2f-4866-b028-9cbbec645100)
```python
predict = lr.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))
print('\n')
print(confusion_matrix(y_test,predict))
```
![image](https://github.com/user-attachments/assets/92e44bc1-51a0-4dfe-adc0-359535967d6d)
![image](https://github.com/user-attachments/assets/8a772a03-b394-4652-aaae-6f5e3a5d9c21)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('classifier', LogisticRegression(random_state=42))  # Logistic Regression model
])

pipeline.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/60267b7d-569d-4ebb-b7ae-04d542fb3cd9)
```python
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test,y_pred))
```
![image](https://github.com/user-attachments/assets/273e7203-36db-45f7-bd19-4228cde8c0db)

## Thanks You







