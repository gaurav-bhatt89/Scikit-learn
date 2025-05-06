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
## Importing K Means cluster from sklearn
from sklearn.cluster import KMeans
kmc = KMeans(n_clusters=4) ## informing the KMeans estimator of how many clusters you want the dataset to be clustered into
kmc.fit(data[0])
kmc.cluster_centers_
fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8,4))
ax1.set_title('K means Cluster')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmc.labels_)
ax2.set_title('Original Cluster')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1])
## Thank You