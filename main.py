import fluctuation_extraction as fe
import pot
import grimshaw
import numpy as np
from data_preprocess import dataloader
import pandas as pd
import smooth

def SD_FluxEV(X, s, p, d, l, k, q):
    n = len(X)
    nt = 0
    a = 2 * s + d + l * (p - 1)
    e, f, S, m = fe.ExtAndSmooth(X, s, p, d, l)

    M = r = y = [None for _ in range(n)]
    thf, t = pot.pot(S[a + 1: a + k], risk = q)
    for Xt in X:
        if Xt > t:
            nt += 1
    for i in range(a + k, n):
        while(i > a + k):
            r[i] = 0
            e[i], f[i], m[i - d], S[i] = smooth.calfeats(X[i - s: i], e, f, m, S, s, i, d, l, p)
            if S[i] > thf:
                r[i] = 1
            elif S[i] > t:
                y[i] = s[i] - t
                # add y[i] in yt
                nt +=  1
                k += 1
                thf, t = pot.pot(y, q, num_candidates=10)
            else:
                k += 1

            if r[i] == 1:
                f[i] = None
                M[i-d] = max(f[i - 2 * d: i])
    return r



if __name__ == '__main__':
    data_path = "../dataset/AIOps2018/decomposed/1th_ts_train.csv"
    X = dataloader(data_path)

    r = SD_FluxEV(X, s, p, d, l, k, q)
