import pandas as pd
import numpy as np

count = 0
#Read CSV file 
rf = pd.read_csv('csvfile.csv', index_col = False, header = 0)
print rf
print rf.values

for i in range(0,len(rf.values)):
    for j in range(0,len(rf[0])):
        if data[i][j].isnull :
            count = count +1
            rf.values[i][j] = 0


##data = np.array(rf)
##print data
##
##for i in range(0,len(data)):
##    for j in range(0,len(data[0])):
##        if data[i][j]. :
##            count = count +1
##            data[i][j] = 0
##print data
##            
