import numpy as np
import matplotlib.pyplot as plt

def lin_reg(x,y):
    '''implements linear regression'''
    ones = np.ones_like(x)    #creating vector of 1s same length as x
    X = np.stack([ones,x],axis=1) #stack 1s and xs to get X matrix 

    w = np.linalg.solve((X.T).dot(X) , (X.T).dot(y)) #compute optimal w

    x_pred = np.linspace(0,1,100) #depends on range and scale of data

    y_pred = (w[1]*x_pred) + w[0]

    plt.plot(x_pred, y_pred, color='black')
    plt.scatter(x, y, marker='x', color='red')
