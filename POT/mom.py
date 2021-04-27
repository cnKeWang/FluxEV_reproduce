import numpy as np

def mom(X):
    n = len(X)
    mu = np.sum(X / n)
    s = np.sum((X - mu)**2 / (n - 1))
    sigma = (mu / 2) * (1 + mu**2 / s**2)
    gamma = 1/2 * (1 - mu**2 / s**2)
    return gamma, sigma