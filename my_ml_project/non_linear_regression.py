import numpy as np
import matplotlib.pyplot as plt
#from my_ml_project.linear_regression_joycelyn import lin_reg, predict_linear 
from my_ml_project.linear_regression_joycelyn import LinearRegression 


class NonLinearRegression(LinearRegression):
    
    def __init__(self, dim=5):
        # class attributes can be accesed by any method (function) whose first argument is self
        self.phi = np.zeros(dim) # 
#         self.lambda = _lambda
        self.dim = dim

    
    def featurize_phi(self, x):
        self.phi = np.array(
            [[x_ ** d for d in range(self.dim + 1)] for x_ in x]
        )
    
        return self.phi

    def fit(self, x_nonlin, y_nonlin):

        self.phi = NonLinearRegression.featurize_phi(x_nonlin)
    
        # abstraction 
        w = LinearRegression.fit(self.phi, y_nonlin, bias=False)
    
        return w


    def predict(self, x_test):
    
    
        return LinearRegression.predict(x_test, bias=False)
    #     return phi.dot(w.T)
    
