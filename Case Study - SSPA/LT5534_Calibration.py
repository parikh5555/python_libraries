## This python code helps to calibrate LT5534 RF Power meter sensor
## Which would help to understand characterisation and to get values of slope
## and offset which further helps to make controller based power meter
## Any this type of application of linear regression is applicable only when
## there is a linear equation

from sklearn import linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = [[]]
y = []

def mean(numbers):
    return float(sum(abs(numbers))) / max(len(numbers), 1)

def readfromcsv():
    rf = pd.read_csv('LT5534_Results.csv', index_col = False, header = 0)
    X = rf [['PCB1']]
    y = rf ['I/P Power (dBm)']
    return X,y

def train_model():
    X,y = readfromcsv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    return X_train, X_test, y_train, y_test, X, y

def Linear_Regression_alg():
    X_train, X_test, y_train, y_test, X, y = train_model()
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train) #Training dataset from all the dataset
    #regressor.fit(X, y) #Whole dataset
    m=regressor.coef_[0] # Slope
    c=regressor.intercept_ #initial condition
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
    print coeff_df  
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
    print df
    from sklearn import metrics  
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
    return X_train, X_test, y_train, y_test, X, y, m, c, y_pred



def visualize():
    X_train, X_test, y_train, y_test, X, y, m, c, y_pred = Linear_Regression_alg()
    plt.scatter (X_test,y_test)
    print len(y_pred), len(X_test)
    plt.plot (X_test,y_pred, 'c')
    
    plt.show()
visualize()
