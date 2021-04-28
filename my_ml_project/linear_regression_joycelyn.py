import numpy as np
import matplotlib.pyplot as plt


# 1. fit w
# 2 predict w
# 3. mse 

#  Homework ? get this to work ?
class LinearRegression(object):
    # This code might not run (try fixing)    
    def __init__(self, dim=1, _lambda =0.001):
        # class attributes can be accesed by any method (function) whose first argument is self
        self.w = np.zeros(dim) # 
#         self.lambda = _lambda
        self.dim = dim
    
    def fit(self, x, y, bias=True):
        """Function that implements linear regression"""
        ones = np.ones_like(x)  # create a vector of 1's with the same length as x

        if bias:
            X = np.stack(
                [ones, x], axis=1
            )  # stack 1's and x's to get the X matrix having the 1's and x's as columns (design matrix for linear regression)
        else:
            X = np.array(x)
        # np.linalg.cho_solve
        self.w = np.linalg.solve(
            (X.T).dot(X) + self.lambbda * np.eye(self.dim), (X.T).dot(y)
        )  # compute the optimal w using the Moore-Penrose pseudoinverse
    
    
    def predict(self, x_test, bias=True):
        ones = np.ones_like(x_test) 
        if bias:
            X = np.stack(
                [ones, x_test], axis=1
            ) 

        y_pred = X.dot(self.w.T) 
        return y_pred

    
# Classes vs functions
    
# # class way (sklearn) 
# model = LinearRegression(dim=1, _lambda=0.001)
# model.fit(x, y, lambda_) # I dont have to pas it lambda
# model.predict(x) # I dont have pass it w 

# #  Function way
# lambda_ = 0.001
# w = lin_reg(x,y, lambda_)
# y_pred = predict_linear(x_test, w)
# w = lin(x,y, lmabga_)
# y_pred_2 = predict_linear(x_test_2, w)
    
        
          

def lin_reg(x, y, bias=True):
    """Function that implements linear regression"""
    ones = np.ones_like(x)  # create a vector of 1's with the same length as x
    
    if bias:
        X = np.stack(
            [ones, x], axis=1
        )  # stack 1's and x's to get the X matrix having the 1's and x's as columns (design matrix for linear regression)
    else:
        X = np.array(x)
        
    # np.linalg.cho_solve
    import pdb; pdb.set_trace()
    w = np.linalg.solve(
        (X.T).dot(X), (X.T).dot(y)
    )  # compute the optimal w using the Moore-Penrose pseudoinverse

    # The above line is equivalent to the following:
    # w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y_lin)
    # but unlike this approach it avoids explicitly computing the matrix inverse, which can be numerically unstable,
    # instead it uses a linear solver, which is more stable. In general, if you don't explicitly need a matrix inverse,
    # you should avoid computing it.

    return w


def predict_linear(x_test, w, bias=True):
    
    # X_test N x D , w 1 x D    X * w.T ( NxD * Dx1) = N x 1 )
    # Phi N x K K >>> D, w = 1xK
    ones = np.ones_like(x_test) 
    if bias:
        X = np.stack(
            [ones, x_test], axis=1
        ) 
    
    y_pred = X.dot(w.T)  # ( (w * X.T).T = X * w.T 
    return y_pred

    
    
def test_lin_regressor():
    x_pred = np.linspace(
        0, 1, 100
    )  # the range depends on the range of data but for an example set 100 points equispaced between 0 and 1

    y_pred = predict(x_pred) # evaluate the linear trendline at the values of x above