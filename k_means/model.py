import pandas as pd
import numpy as np

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
data['sepal_width']=data['sepal_width'].where(data['sepal_width']>upper_limit, data['sepal_width'].mean())
data['sepal_width']=data['sepal_width'].where(data['sepal_width']<lower_limit, data['sepal_width'].mean())
data = data.drop_duplicates()
#save the data to a csv file
data.to_csv('preprocessed.csv')
df=pd.read_csv('preprocessed.csv')
print(df)
print('The data with the sepal width higher than the upper limit is ', df.loc[df['sepal_width']>upper_limit])
print('The data with the sepal width lower than the lower limit is ', df.loc[df['sepal_width']<lower_limit])
