import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# Based on their nature, machine learning models suck at extrapolation,
# the range of the given data is the only range where the model can predict
# accurately. To illustrate this, we can use a simple example of a 2D function,
# where the model is trained on a range of -3 to 3, and we want to predict
# the function outside this range. The model will fail to predict the function
# accurately outside the range it was trained on. This is a common problem in
# machine learning, and it is important to be aware of it when using
# machine learning models for prediction tasks.

# ===> To prevent this to happen, neural networks is a good choice.
# Neural networks are universal function approximators, which means
# they can approximate any function given enough data and computational resources.
# This is why neural networks are used in many applications where the function
# to be approximated is unknown or too complex to model using traditional
# machine learning models.

N = 100
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2*X[:, 0]) + np.cos(3*X[:, 1])


def plot_surface(model, X, Y):
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y)

    line = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(line, line)
    Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
    Yhat = model.predict(Xgrid).flatten()
    ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat,
                    linewidth=0.2, antialiased=True)

    plt.show()


if __name__ == '__main__':

    # # !!!! Test different models to see the difference

    # model = SVR(C=100.)
    # model = RandomForestRegressor()
    model = MLPRegressor(hidden_layer_sizes=128, alpha=0.,
                         learning_rate_init=0.01)
    model.fit(X, Y)

    plot_surface(model, X, Y)
