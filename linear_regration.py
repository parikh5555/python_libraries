## simple linear regression a single independent variable
## is used to predict the value of a dependent variable.
## 10 persons height and weight combinations are given
## Linear regration helps to predict 11th person weight from height
## Y = mX + C where m = Slope, C = Initial condition
## This formula is for understading the concepet
## let x1 = mean(x) = sum(x[i])/len(x) and y1 = mean(y) = sum(y[i])/len(y)
## Slope m = sum((x1-x[i])(y1-y[i])) /sum(x1-x[i])^2 For i in range (0,len(x))
## Initital condition C = y1 - m*x1

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
c=reg.intercept_ #initial condition
print("slope=",m, "intercept=",c)

plt.scatter(height,weight,color='black')
predicted_values = [reg.coef_ * i + reg.intercept_ for i in height]
plt.plot(height, predicted_values, 'c') # linear plot
plt.xlabel("height")
plt.ylabel("weight")
plt.show()
#print len(predicted_values)

## Accuracy of the model
## Training accuracy : check y from few elements of x from the database
## Out of sample accuracy : Check y from out of dataset value of x -> should be high
## Model of evaluation
## 1. Test on the portion of dataset which is used to train it
## 2. Training and testing dataset should be mustually exclusive
## 2 -> gives better accuracy for real world problem

## Error = 1/n(Sum(Y(actual value) - Y1(Predicated values)

actual_weight = weight[-4:]
predicated_values_input_height = np.array([[6.1],[6.2],[6.4],[6.8]])
predicated_values_output_weight =  np.zeros(4)

predicted_values_output_weight = (m * predicated_values_input_height )+c
print "Prediction results", predicted_values_output_weight

error_in_prediction = np.zeros(4)
for i in range (0,len(predicted_values_output_weight)):
    error_in_prediction[i] = actual_weight[i]-predicted_values_output_weight[i][0]
print "Error in prediction", error_in_prediction

## Mean absoulute error = Mean(abs(error))
## Mean Square error = Mean(square(error)) -> Increase exponantially
## so increse for larger error with respect to smaller once
## Root mean square error = Root(mean(square(error)))
## Relative absolute error = Sum(error)/sum(Y(predicted)-Y'(mean))
## Relative sqaure error = sum(squre(error))/sum(squre(Y(predicted)-Y'(mean)))
## R = root(1 - Relative sqaure error) -> gives how close data point to predicted value
mean_abs_err = (1.0/len(error_in_prediction))*(sum(abs(error_in_prediction)))
print mean_abs_err, "MAE"

mean_sqaure_error = (1.0/len(error_in_prediction))*(sum(error_in_prediction*error_in_prediction))
print mean_sqaure_error, "MSE"

root_mean_square_error = mean_square_error**(0.5)
print root_mean_square_error, "RMSE"


