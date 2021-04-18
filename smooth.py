import pandas as pd
import numpy as np


def calfeats(X, e, f, m, S, s, i, d, l, p):

    data = pd.DataFrame(X)
    k_ewma = data.ewm(span=s - 1).mean()
    e[i] = X[i] - k_ewma
    delta_sigma = np.var(e[i - s: i]) - np.var(e[i - s: i - 1])
    f[i] = max(delta_sigma, 0)
    f_period = f[i - 2 * d: i]
    m[i - d] = max(f_period)
    delta_fi = f[i] - max(m[i - l * (p - 1): i - l])
    S[i] = max(delta_fi, 0)

    return e[i], f[i], S[i], m[i - d]