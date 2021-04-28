import numpy as np
import matplotlib.pyplot as plt
from my_ml_project.linear_regression_joycelyn import lin_reg, predict_linear 



def featurize_phi(x, D=5):
    phi = np.array(
        [[x_ ** d for d in range(D + 1)] for x_ in x]
    )
    
    return phi


def non_lin_reg(x_nonlin, y_nonlin):

    phi = featurize_phi(x_nonlin)
    
    # abstraction 
    w = lin_reg(phi, y_nonlin, bias=False)
    
    return w


def predict(x_test, w):
    
    phi = featurize_phi(x_test)
    
    return predict_linear(phi, w, bias=False)
#     return phi.dot(w.T)
    
