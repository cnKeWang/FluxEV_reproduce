import pandas as pd
import numpy as np
from CalcEWMA import calculate
from max_l import max_l

def calfeats(X, E, f, m, S, s, i, d, l, p):
    data = X.tolist()
    k_ewma = calculate(data, i, s, alpha=0.5)
    E[i] = X[i] - k_ewma
    delta_sigma = np.var(E[i - s: i]) - np.var(E[i - s: i - 1])

    f[i] = max(delta_sigma, 0)
    store = []
    store.append(f[i])
    m[i - d] = max(store)
    delta_fi = f[i] - max_l(m[i - (l * (p - 1)): i - l], l, p)
    S[i] = max(delta_fi, 0)

    return E[i], f[i], S[i], m[i - d]