import numpy as np
import matplotlib.pyplot as plt


def lin_reg(x, y):
    """Function that implements linear regression"""
    ones = np.ones_like(x)  # create a vector of 1's with the same length as x

    X = np.stack(
        [ones, x], axis=1
    )  # stack 1's and x's to get the X matrix having the 1's and x's as columns (design matrix for linear regression)

    w = np.linalg.solve(
        (X.T).dot(X), (X.T).dot(y)
    )  # compute the optimal w using the Moore-Penrose pseudoinverse

    # The above line is equivalent to the following:
    # w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y_lin)
    # but unlike this approach it avoids explicitly computing the matrix inverse, which can be numerically unstable,
    # instead it uses a linear solver, which is more stable. In general, if you don't explicitly need a matrix inverse,
    # you should avoid computing it.

    x_pred = np.linspace(
        0, 1, 100
    )  # the range depends on the range of data but for an example set 100 points equispaced between 0 and 1

    y_pred = (w[1] * x_pred) + w[
        0
    ]  # evaluate the linear trendline at the values of x above

    plt.plot(x_pred, y_pred, color="black")  # plot the trendline
    plt.scatter(x, y, marker="x", color="red")  # plot the datapoints
    plt.show()
    print(w[1], w[0])
