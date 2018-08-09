## simple linear regression a single independent variable
## is used to predict the value of a dependent variable.
## 10 persons height and weight combinations are given
## Linear regration helps to predict 11th person weight from height
## Y = mX + C where m = Slope, C = Initial condition
## This formula is for understading the concepet
## let x1 = mean(x) = sum(x[i])/len(x) and y1 = mean(y) = sum(y[i])/len(y)
## Slope m = sum((x1-x[i])(y1-y[i])) /sum(x1-x[i])^2 For i in range (0,len(x))
## Initital condition C = y1 - m * x1

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

height = np.array([[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]])
weight = np.array([42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69])

plt.scatter(height, weight) #scatter plot
plt.xlabel("height")
plt.ylabel("weight")
#plt.show()

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(height,weight)

#the coefficients 
m=reg.coef_[0] # Slope
b=reg.intercept_ #initial condition
print("slope=",m, "intercept=",b)

plt.scatter(height,weight,color='black')
predicted_values = [reg.coef_ * i + reg.intercept_ for i in height]
plt.plot(height, predicted_values, 'b') # linear plot
plt.xlabel("height")
plt.ylabel("weight")
plt.show()
print len(predicted_values)
