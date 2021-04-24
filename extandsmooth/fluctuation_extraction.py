import pandas as pd
import numpy as np
from extandsmooth.CalcEWMA import calculate
from max_l import max_l

def ExtAndSmooth(X, s, p, d, l):
    '''

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
    E = [None for _ in range(s)]
    f = [None for _ in range(2 * s)]
    m = [None for _ in range(2 * s + d)]

    S = [None for _ in range(2 * s + d + l * (p - 1))]
    data = X.tolist()
    for i in range(n):
        if i > s - 1:
            # 计算EWMA
            k_ewma = calculate(data, i, s, alpha=0.5)
            Ei = data[i] - k_ewma
            E.append(Ei)
        if i > (2 * s - 1):
            delta_sigma = np.var(E[i - s: i]) - np.var(E[i - s: i - 1])
            fi = max(delta_sigma, 0)
            f.append(fi)
        if i > (2 * s + 2 * d - 1):
            f_period = f[i - 2 * d: i]
            m_i_d = max(f_period)
            m.append(m_i_d)
        if i > 2 * s + d + l * (p - 1) - 1:
            delta_fi = f[i] - max_l(m[i - (l * (p - 1)): i - l], l, p)
            Si = max(delta_fi, 0)
            S.append(Si)

    return E, f, m, S
