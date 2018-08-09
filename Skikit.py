from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

height = [[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
weight = [42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69]

plt.scatter(height, weight)
plt.xlabel("height")
plt.ylabel("weight")
