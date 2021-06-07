import pandas as pd
import numpy as np
from extandsmooth.CalcEWMA import calculate
from max_l import max_l

def ExtAndSmooth(X, s, p, d, l):
    '''
    波动提取和两步平滑
    :param X: Time series data
    :param s: window sizes
    :param p: p period
    :param d: half of the window sizes to handle data drift
    :param l: period length
    :return:
        E: the prediction error
        f: the fluctuation value after the first smoothing
        m: the local maximum of f
        S: the fluctuation value after the second smoothing
    '''

    n = len(X)
    E = [None for _ in range(n)]
    f = [None for _ in range(n)]
    m = [None for _ in range(n - d)]

    S = [None for _ in range(n)]
    data = X.tolist()
    for i in range(n):
        if i > s - 1:
            # 计算EWMA
            k_ewma = calculate(data, i, s, alpha=0.4)
            E[i] = data[i] - k_ewma

        if i > (2 * s - 1):
            delta_sigma = np.var(E[i - s: i + 1]) - np.var(E[i - s: i])
            f[i] = max(delta_sigma, 0)

        if i > (2 * s + 2 * d - 1):
            f_period = f[i - 2 * d: i + 1]
            m[i - d] = max(f_period)

        if i > 2 * s + d + l * (p - 1) - 1:
            delta_fi = f[i] - max_l(m[i - (l * (p - 1)): i - l + 1], l, p)
            S[i] = max(delta_fi, 0)


    return E, f, m, S
