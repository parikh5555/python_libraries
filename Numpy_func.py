import numpy as np

## array [x:y] where x is index from which array needs to start and y is till
## when array is present



# Create numpy single dimention array 
def create_1_dim_array():
    a = np.array([1, 2, 3])   # Create a rank 1 array
    #print(type(a))            # Prints "<class 'numpy.ndarray'>"
    #print(a.shape)            # Prints "(3,)"
    #call array by element same as array
    print a[0], a[1], a[2]    # Prints "1 2 3"
    #change array element by index 
    a[0] = 5                  # Change an element of the array
    print a                   # Prints "[5, 2, 3]"

def create_2_dim_array():
    b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
    #print(b.shape)                     # Prints "(2, 3)"
    #print b
    #array indexing
    print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"

def create_blank_arrays(n,m,k):
    #n*m array with K value if K is not used shoulb be left 0
    a = np.zeros((n,m))  #create zeros of n*m
    #a = np.ones((n,m)) #create ones of n*m
    #a = np.full((n,m),k) #create k element of n*m
    #a = np.empty((n,m))  #create empty of n*m
    print a

def create_multi_dim_array():
    a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    print a
    # array [ limit rows, limit columns]
    # Cropping of array
    subarr = a[:2, 1:3]
    #print subarr
    # Accessing a row of array
    row = a[1]    # or c[1,:]
    #print row
    # Accessing a column
    column = a[: , 1]
    #print column
    # Element of array array[row, cloumn]
    #print(a[0, 1])   # Prints "2"
    # change element of array
    #b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
    #print(a[0, 1])   # Prints "77"
    
create_multi_dim_array()
