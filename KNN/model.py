import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data=pd.read_csv('Social_Network_Ads.csv', index_col=0)

#Encode the gender column
y_gender=pd.get_dummies(data['Gender'], prefix='Gender', drop_first=True).astype(int)
data=pd.concat([data, y_gender], axis=1).drop('Gender', axis=1)
print(f'The ENCODED DATA is: \n', data)
print('=================================================')

#scale the data
sc=StandardScaler()
descrete_columns=data[['Purchased', 'Gender_Male']]
data1=data.drop(columns=['Purchased','Gender_Male'])
scaled_data=sc.fit_transform(data1)
data2=pd.DataFrame(scaled_data,columns=data1.columns)
data=pd.concat([data2,descrete_columns.reset_index(drop=True)], axis=1)
print('SCALED DATA\n', data, '\n============================================')

#FEATURE SELECTION
print(f'The columns in the data are: {data.columns}' '\n=======================================================')
y=data['Purchased']
x=data.drop('Purchased', axis=1)
 
print(f'Shape of x is {x.shape}\n',x)
print(f'Shape of y is {y.shape}\n',y)
print('============================')
#Split the train and test
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.30,random_state = 42)
print(f'Shape of x_train is {x_train.shape}')
print(f'Shape of y_train is {y_train.shape}')
print(f'Shape of x_test is {x_test.shape}')
print(f'Shape of y_test is {y_test.shape}')
print('============================')



