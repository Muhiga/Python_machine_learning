import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



#FEATURE SELECTION
data=pd.read_csv('Advertising.csv', index_col=0)
x=data.drop('Sales', axis=1)
y=data['Sales']
print(f'Shape of x is {x.shape}')
print(f'Shape of y is {y.shape}')
print('============================')
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.30,random_state = 42)
print(f'Shape of x_train is {x_train.shape}')
print(f'Shape of y_train is {y_train.shape}')
print(f'Shape of x_test is {x_test.shape}')
print(f'Shape of y_test is {y_test.shape}')
print('============================')

model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#model performance
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print(f'The Mean squared error is {mse}')
print(f'The Mean absolute error is {mae}')
print(f'The root mean squared error is {rmse}')
print(f'The r squared is {r2}')

