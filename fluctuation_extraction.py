import pandas as pd
import numpy as np


def ExtAndSmooth(X, s, p, d, l):
    n = len(X)
    e = f = S = [None for _ in range(n)]
    m = [None for _ in range(n - d)]
    for i in range(n):
        e[i] = f[i] = S[i] = m[i] = None
        if i > s:
            # 计算EWMA
            data = pd.DataFrame(X)
            k_ewma = data.ewm(span=s - 1).mean()
            e[i] = X[i] - k_ewma
        if i > 2 * s:
            delta_sigma = np.var(e[i - s: i]) - np.var(e[i - s: i - 1])
            f[i] = max(delta_sigma, 0)
        if i > 2 * s + 2 * d:
            f_period = f[i - 2 * d: i]
            m[i - d] = max(f_period)
        if i > 2 * s + d + l * (p - 1):
            delta_fi = f[i] - max(m[i - l * (p - 1): i - l])
            S[i] = max(delta_fi, 0)

    return e, f, S, m
