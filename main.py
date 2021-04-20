import fluctuation_extraction as fe
import pot
import grimshaw
import numpy as np
from data_preprocess import dataloader
import pandas as pd
import smooth

def SD_FluxEV(X, p, s, d, l, k, risk):
    risk = 0.0004
    n = len(X)
    nt = 0
    a = 2 * s + d + l * (p - 1)
    e, f, S, m = fe.ExtAndSmooth(X, s, p, d, l)

    M = r = y = [None for _ in range(n)]
    print("S:", S)
    thf, t = pot.pot(np.array((S[a + 1: a + k])), risk = risk)
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
                thf, t = pot.pot(y, risk=risk, num_candidates=10)
            else:
                k += 1

            if r[i] == 1:
                f[i] = None
                M[i-d] = max(f[i - 2 * d: i])
    return r


data_path = "../dataset/AIOps2018/decomposed/1th_ts_train.csv"
timestamp, value = dataloader(data_path)
X = pd.DataFrame(value)



s = 240
p = 2
d = 120
l = 87000
r = SD_FluxEV(X=value, s=240, p=2, d=120, l=87000, k=12, risk=1e-4)
