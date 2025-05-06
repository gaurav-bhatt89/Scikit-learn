# This below Python code executed in Jupyter Notebook by [Anaconda Navigator](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.anaconda.com/products/navigator&ved=2ahUKEwiT5K_m_IuNAxWce_UHHVooNSwQFnoECBkQAQ&usg=AOvVaw2FiVm4Knmhe7xplbtYwLdO) 
## We will learn to 
  1. Generate synthetic dataset using make_blobs on sklearn
  2. Pass the data through K Means estimator to segregrate the data into required clusters 

### GitHub Notebook - [Link](https://github.com/gaurav-bhatt89/Scikit-learn/blob/main/K_Means_Clustering_Blob_Tables.ipynb)
### NBViewer - [Link](https://nbviewer.org/github/gaurav-bhatt89/Scikit-learn/blob/main/K_Means_Clustering_Blob_Tables.ipynb)
```python
## Lets import necessary python libraries
import numpy as np # Numerical Python library used to perform numerical operations
import pandas as pd # Pandas library used to handle and analyze structured data
import matplotlib.pyplot as plt # Plotting library
import seaborn as sns # Additional plotting library built over matplotlib (better visuals)
import string
sns.set_theme(style='darkgrid',palette='Set2')

%matplotlib inline
from sklearn.datasets import make_blobs #generating synthetic datasets, particularly useful for clustering and classification algorithm testing

## Lets make a blob of four centers with a STD of 1.5

data = make_blobs(n_samples=200,n_features=2,centers=4,shuffle=True,cluster_std=1.5)

## plotting the generated data set with actuals 4 centres
plt.figure(figsize=(6,4))
plt.scatter(data[0][:,0],data[0][:,1],c=data[1])
plt.title('Actual samples with 4 centres colorred')
plt.show()
```
![image](https://github.com/user-attachments/assets/f6617391-c4d3-44eb-9807-d0c2f4f237b4)
```python
## Importing K Means cluster from sklearn
from sklearn.cluster import KMeans

kmc = KMeans(n_clusters=4) ## informing the KMeans estimator of how many clusters you want the dataset to be clustered into

kmc.fit(data[0])
```
![image](https://github.com/user-attachments/assets/9b6efea8-e437-49ef-b9b2-ce2dd74cfd0d)
```python
kmc.cluster_centers_
array([[ 7.50931207,  8.69139956],
       [ 7.78406086, -3.27992931],
       [-4.50559933,  8.00838877],
       [ 3.2115619 ,  6.51357994]])
fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8,4))

ax1.set_title('K means Cluster')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmc.labels_)

ax2.set_title('Original Cluster')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1])

plt.show()
```
![image](https://github.com/user-attachments/assets/9ea2f330-eaa4-4d85-a555-8574e823b2e3)

## Thank You

