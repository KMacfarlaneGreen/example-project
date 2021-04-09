import numpy as np
import matplotlib as plt


def non_lin_reg(x_nonlin, y_nonlin):

    D = 5  # D represents the order of the polynomial

    phi = np.array(
        [[x_ ** d for d in range(D + 1)] for x_ in x_nonlin]
    )  # build the design matrix Phi

    # compute the optimal w using the Moore-Penrose pseudoinverse
    w = np.linalg.solve((phi.T).dot(phi), (phi.T).dot(y_nonlin))

    # As with linear regression, the line above is numerically stable version of line commented line below (e.g. it works with D>N)
    # w = np.linalg.inv((phi.T).dot(phi)).dot(phi.T).dot(y_nonlin) # apply the Moore-Penrose pseudoinverse using Phi

    x_pred = np.linspace(0, 1, 100)  # 100 points equispaced between 0 and 1

    phi_pred = np.array(
        [[x_ ** d for d in range(D + 1)] for x_ in xs]
    )  # design matrix for points at which to plot

    y_pred = phi_pred.dot(w)  # output of the model at the points above

    ######## Plotting code #######
    print(
        "Sum squared errors for polynomial of order {}:".format(D),
        np.sum((phi.dot(w) - y_nonlin) ** 2).round(3),
    )
    plt.scatter(x_nonlin, y_nonlin, marker="x", color="red")  # plot model predictions
    plt.plot(x_pred, y_pred, color="black")  # plot dataset

    plt.show()
