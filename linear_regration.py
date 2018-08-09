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
