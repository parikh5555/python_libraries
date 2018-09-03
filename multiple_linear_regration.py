## multiple linear regression a multiple independent variable
## is used to predict the value of a dependent variable.
## 10 persons height, age and gender and weight  combinations are given
## Linear regration helps to predict 11th person weight from alll above 
## Y = m1X1 + m2X2 + ...+mNXN + C0 where m = Slope, C = Initial condition
## This formula is for understading the concepet
## 

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt


dataset = pd.read_csv('C:\Users\Lenovo\Desktop\Machine Learning\Python\petrol_consumption.csv')

## pandas dataset head function gets first five rows
#print dataset.head()

##
#print dataset.describe()

X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways',  
       'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']

#print X
#print y

## Train test split function will split training dataset with testing dataset 
## train_test_split(input, output, 20% test size and 80% training size, random state)
## Instead of random all the set should be chosen as a training set in real life
##


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

#print X_train, X_test
#print Y_train, Y_test

## fit function (training input, training output)
## 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #Training dataset from all the dataset
#regressor.fit(X, y) #Whole dataset
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print coeff_df  
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print df

#for details of Dataframe function check pandas library

#for MEA, MSE, RMSE check simple linear regression

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
