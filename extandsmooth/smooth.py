import pandas as pd
import numpy as np
from extandsmooth.CalcEWMA import calculate
from max_l import max_l

def calfeats(X, E, f, m, S, s, i, d, l, p):
    '''


    对X进行平滑操作
    '''
    data = X.tolist()
    k_ewma = calculate(data, i, s, alpha=0.4)
    E[i] = X[i] - k_ewma
    delta_sigma = np.var(E[i - s: i + 1]) - np.var(E[i - s: i])

    f[i] = max(delta_sigma, 0)
    f_period = f[i - 2 * d: i + 1]
    m[i - d] = max(f_period)
    delta_fi = f[i] - max_l(m[i - (l * (p - 1)): i - l + 1], l, p)
    S[i] = max(delta_fi, 0)

    return E[i], f[i], S[i], m[i - d]