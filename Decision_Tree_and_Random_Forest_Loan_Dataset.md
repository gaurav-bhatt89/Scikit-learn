# This below Python code executed in Jupyter Notebook by [Anaconda Navigator](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.anaconda.com/products/navigator&ved=2ahUKEwiT5K_m_IuNAxWce_UHHVooNSwQFnoECBkQAQ&usg=AOvVaw2FiVm4Knmhe7xplbtYwLdO) 
## We will learn to 
  1. Load an existing publicly available ['Loan Paid'](https://github.com/gaurav-bhatt89/Datasets/blob/main/Loan_Data_DT_RF.csv) dataset
  2. Exploratory Data Analysis
  3. Data Cleansing / Wrangling
  4. Feature Consolidation / Engineering
  5. Applying Decision Tree estimator and evaluating its predction accuracy
  6. Applying Random Forest estimator and evaluating its prediction accuracy

### GitHub Notebook - [Link](https://github.com/gaurav-bhatt89/Scikit-learn/blob/main/Decision_Tree_and_Random_Forest_Loan_Dataset.ipynb)
### NBViewer - [Link](https://nbviewer.org/github/gaurav-bhatt89/Scikit-learn/blob/main/Decision_Tree%2BRandom_Forest_Loan_Dataset.ipynb)
```python
## Lets import necessary python libraries
import numpy as np # Numerical Python library used to perform numerical operations
import pandas as pd # Pandas library used to handle and analyze structured data
import matplotlib.pyplot as plt # Plotting library
import seaborn as sns # Additional plotting library built over matplotlib (better visuals)
sns.set_theme(style='darkgrid',palette='viridis')
```
```python
%matplotlib inline
import string # This library will help us perform operations on string values
```
```python
## Lets import the loan data set and load iti nto a dataframe 
df = pd.read_csv(r"C:\Users\gmraj\Documents\Refactored_Py_DS_ML_Bootcamp-master\15-Decision-Trees-and-Random-Forests\loan_data.csv")
```
```python
df.info()
```
![image](https://github.com/user-attachments/assets/9cb7bf6d-1ea2-4326-8fa5-38ce8067b950)
```python
## Exploratory Data Analysis
## Data Wrangling, Feature Consolidation/Engineering

## Checking for missing values
plt.figure(figsize=(6,4))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title('Missing values will be highlighted by a yellow underline')
plt.show()
```
![image](https://github.com/user-attachments/assets/073a1d9f-b877-4f65-a9d9-878919b5abf0)
```python
## No missing value found

## Lets check for correlation of the independent variables
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('Correlation Coefficient of Variables')
plt.show()
```
![image](https://github.com/user-attachments/assets/7bec76ab-9132-414f-8d42-28fe3339cf4c)
```python
sns.scatterplot(x='fico',y='int.rate',data=df)
plt.title('FICO Score vs Int. Rate [Correlation Coefficient Value: -0.71]')
plt.show()
```
![image](https://github.com/user-attachments/assets/489b4a0e-b408-4e85-8007-ff8b7ed43d76)
```python
## As observed from the Heatmap, there is negative correlation with the coefficient value of -0.71 between FICO and Int. Rate
## As the FICO goes up, the Int. Rate goes down and vice verca

df.head(3)
```
![image](https://github.com/user-attachments/assets/7e81b61a-6511-4060-9497-5c9048e7f819)
```python
## Lets create a new dataframe by tweaking existing one in such a way that we obtain
## purpose wise percentage of all the loans which are not fully paid

purpose_groupby = df.groupby('purpose')['not.fully.paid'].describe()
purpose_groupby = pd.DataFrame(purpose_groupby)

purpose_groupby = purpose_groupby.sort_values(by='mean',ascending=True)

purpose_groupby
```
![image](https://github.com/user-attachments/assets/520bf965-25b3-469b-afb6-de0a202dfb9c)
```python
## Here we can see that 'small_busines' category has highest amount of loans which are not fully paid
## where as 'major_purchase' and 'credit_card' has the highest amount of loans which are fully paid

## Lets now feature engineer the existing dataframe so it becomes ML algorithm ready

## We will have to encode all the categorical variables so that all the data in the dataframe is in numerical values

purpose = pd.get_dummies(df['purpose'],drop_first=True,dtype='float')
df.drop('purpose',axis=1,inplace=True)

df = pd.concat([df,purpose],axis=1)

df.head(5)
```
![image](https://github.com/user-attachments/assets/938cf6b6-b654-4fc6-8252-6d71a7ccbe14)
```python
## Standardize the dataset

from sklearn.model_selection import train_test_split # to split the dataframe into a training and a testing set
from sklearn.preprocessing import StandardScaler # to standardize the dataframe values so that the values are standarized on a single x axis
from sklearn.tree import DecisionTreeClassifier # Decision Tree Algorithm (estimator)
from sklearn.pipeline import Pipeline # to automate the steps

X = df.drop('not.fully.paid',axis=1)

y = df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

## Creating a pipeline that will first standardize the values and the pass them through the Decision Tree estimator
pipes = Pipeline([
    ('scaler',StandardScaler()),
    ('dec_tree',DecisionTreeClassifier())
])

pipes.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/03ffb8fd-05ae-4eac-94a7-361318430db8)
```python
## Importing metrics required for performance evaluation
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

predict_pipe = pipes.predict(X_test)

print(classification_report(y_test,predict_pipe))
print('\n')
print(confusion_matrix(y_test,predict_pipe))
```
![image](https://github.com/user-attachments/assets/eedb1454-0357-421b-af25-7e931df35ea1)

![image](https://github.com/user-attachments/assets/a013ab84-0e1d-499d-b5a4-600adc589589)
```python
from sklearn.ensemble import RandomForestClassifier

pipes_rf = Pipeline([
    ('scaler_rf',StandardScaler()),
    ('rc_classifier',RandomForestClassifier(n_estimators=200))
])

pipes_rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/e3c13023-fe46-4e9d-b6c4-50fc2404e215)
```python
predict_rf = pipes_rf.predict(X_test)

print(classification_report(y_test,predict_rf))
print('\n')
print(confusion_matrix(y_test,predict_rf))
```
![image](https://github.com/user-attachments/assets/85dbaa43-f56d-4281-bc24-c355408b9770)

## Thank You













