import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score, classification_report

data=pd.read_csv('diabetes.csv')

#Replace the values with zeros with the medium of the column. The values of glucose, bloodpressure, insulin, BMI and skin thickness can not be zero.
columns_to_replace=['Glucose','BloodPressure','Insulin','BMI','SkinThickness']
for column in data.columns:
    median=data[column].median()
    if column in columns_to_replace:
       data[column]=data[column].replace(0,median)
   # Handle the outliers
    Q1=data[column].quantile(0.25)
    Q3=data[column].quantile(0.75)
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR
    upper_limit=Q3+1.5*IQR
    data[column] = np.where((data[column] < lower_limit) | (data[column] > upper_limit), median, data[column])
#Push the preprocessed data into a csv file and     
data.to_csv('preprocessed.csv', index=False)
preprocessed=pd.read_csv('preprocessed.csv')
print('This is the preprocessed data')
print(preprocessed)
print('=====================================')

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
columns_to_scale = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
data[columns_to_scale]=scaler.fit_transform(data[columns_to_scale])

#Split the data into the  train and test
x=preprocessed.drop('Outcome',axis=1)
y=preprocessed['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print('======================================') 

#Fitting the model
model=LogisticRegression(max_iter=10000, verbose=1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#Evaluating the model
conf_matrix=confusion_matrix(y_test,y_pred)
accuracy_score = accuracy_score(y_test,y_pred)
precision_score=precision_score(y_test,y_pred)
recall_score=recall_score(y_test,y_pred)
f1_score=f1_score(y_test,y_pred)
print('The confusion matrix is:')
print(conf_matrix)
print(f'The accuracy score is {accuracy_score}')
print(f'The precision score {precision_score}')
print(f'The recall score is {recall_score}')
print(f'the F1 score is {f1_score}')
print('Classification Report')
print(classification_report(y_test,y_pred))
print('=========================================')
