import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

import warnings
warnings.filterwarnings('ignore')

#Preprocessing
data1 = pd.read_csv('Iris.csv', index_col=0)
#drop the duplicates
data = data1.drop_duplicates()
#handle the outliers in the sepal_width column. The data has  a normal distribution and hence we use  the 3 sigma method
lower_limit=data['sepal_width'].mean()-3*data['sepal_width'].std()
upper_limit=data['sepal_width'].mean()+3*data['sepal_width'].std()

#find the percentage of the outliers. 
print('The percentage of data below the lower limit is: ',len(data.loc[data['sepal_width']<lower_limit])/len(data)*100)
print('The percentage of data above the upper limit is: ',len(data.loc[data['sepal_width']>upper_limit])/len(data)*100)

print('The original data with the sepal width higher than the upper limit is ', data.loc[data['sepal_width']>upper_limit])
print('The original data with the sepal width lower than the lower limit is ', data.loc[data['sepal_width']<lower_limit])

data.loc[data['sepal_width']>upper_limit,'sepal_width']=data['sepal_width'].mean()

data = data.drop_duplicates()
#save the data to a csv file
data.to_csv('preprocessed.csv')
df=pd.read_csv('preprocessed.csv')
print(df)
print('The preprocessed data with the sepal width higher than the upper limit is ', df.loc[df['sepal_width']>upper_limit])
print('The preprocessed data with the sepal width lower than the lower limit is ', df.loc[df['sepal_width']<lower_limit])

#Scaling
scaling=StandardScaler()

data[['sepal_width', 'sepal_length','petal_length', 'petal_width']]=scaling.fit_transform(data[['sepal_width', 'sepal_length','petal_length', 'petal_width']])
print('--------SCALED DATA--------- ')
print(df)

data = data.drop_duplicates()

#Scaling
scaling=StandardScaler()
data[['sepal_width', 'sepal_length','petal_length', 'petal_width']]=scaling.fit_transform(data[['sepal_width', 'sepal_length','petal_length', 'petal_width']])

#Remove the target
x=data.drop('target',axis=1)

#use the elbow method to determine the optimal number of clusters fro the kmeans clustering

wcss=[]
for i in range(1,11):
	model=KMeans(n_clusters=i, init='k-means++',max_iter=300, n_init=10, random_state=0)
	model.fit(x)
	wcss.append(model.inertia_)
#plot a graphof the wcss vs the number of clusters
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.ylabel('WCSS')
plt.xlabel('The number of clusters')
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/the_elbow_method.png')
plt.close() 
#implementing the K-Means clustering
model=KMeans(n_clusters=3, init='k-means++', max_iter=300,n_init=10, random_state=0)
y_pred=model.fit_predict(x)

#Visualising the clusters
plt.scatter(x.loc[y_pred == 0, x.columns[0]], x.loc[y_pred == 0, x.columns[1]], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x.loc[y_pred == 1, x.columns[0]], x.loc[y_pred == 1, x.columns[1]], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x.loc[y_pred == 2, x.columns[0]], x.loc[y_pred == 2, x.columns[1]], s = 100, c = 'green', label = 'Iris-virginica')
#Plotting the centroids of the clusters
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.legend()
plt.savefig('plots/cluster_visualizations')
plt.close()
# 3d scatterplot using matplotlib

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x.loc[y_pred == 0, x.columns[0]], x.loc[y_pred == 0, x.columns[1]], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x.loc[y_pred == 1, x.columns[0]], x.loc[y_pred == 1, x.columns[1]], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x.loc[y_pred == 2, x.columns[0]], x.loc[y_pred == 2, x.columns[1]], s = 100, c = 'green', label = 'Iris-virginica')
#Plotting the centroids of the clusters
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.savefig('plots/3d_cluster_visualization')

#get the silhoutte score
print(f'The Silhouette score is:\n{silhouette_score(x, model.labels_)}')
